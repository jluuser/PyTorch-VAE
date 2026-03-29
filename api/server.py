#!/usr/bin/env python3
"""
Single-GPU AEOT generation API (queue + worker).

Usage:
  cd /home/zky/PyTorch-VAE
  export CUDA_VISIBLE_DEVICES=0
  uvicorn api.server:app --host 0.0.0.0 --port 8000

Notes:
- This server is intentionally single-worker (one job at a time) to avoid GPU contention.
- It wraps scripts/run_aeot_end2end.py and returns task status + output paths.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


REPO_ROOT = Path("/home/zky/PyTorch-VAE")
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_aeot_end2end.py"
DEFAULT_AE_CONFIG = str(REPO_ROOT / "configs" / "stage1_ae.yaml")
DEFAULT_AE_CKPT = "/home/zky/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt"
DEFAULT_FEATURES_PT = "/home/zky/AE-OT/results_curves/features_5w.pt"
DEFAULT_OT_H = "/home/zky/AE-OT/results_curves/h.pt"
DEFAULT_OT_ROOT = "/home/zky/AE-OT"
DEFAULT_OUT_ROOT = "/data/zky/api_results"
DEFAULT_GPU_ID = 0


class GenerateRequest(BaseModel):
    ae_ckpt: str = Field(DEFAULT_AE_CKPT, description="Absolute path to AE checkpoint")
    n_generate: int = Field(1000, ge=1)
    num_gen_x: int = Field(50000, ge=1000)
    ot_bat_size_n: int = Field(10000, ge=1)
    ot_thresh: float = Field(0.3)
    decode_batch_size: int = Field(128, ge=1)
    seed: int = Field(42)
    run_name: Optional[str] = None

    # Optional overrides
    ae_config: str = DEFAULT_AE_CONFIG
    features_pt: str = DEFAULT_FEATURES_PT
    ot_h: str = DEFAULT_OT_H
    ot_root: str = DEFAULT_OT_ROOT
    out_root: str = DEFAULT_OUT_ROOT
    gpu_id: int = DEFAULT_GPU_ID


@dataclass
class TaskState:
    task_id: str
    status: str = "queued"  # queued/running/done/failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    run_dir: str = ""
    summary_path: str = ""
    stdout_tail: str = ""
    error: str = ""


app = FastAPI(title="AEOT Single-GPU API", version="1.0.0")
WEB_DIR = REPO_ROOT / "api" / "web"
if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

task_queue: "queue.Queue[tuple[str, GenerateRequest]]" = queue.Queue()
tasks: Dict[str, TaskState] = {}


def _sanitize_run_name(name: str) -> str:
    name = (name or "").strip()
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("._-")
    return out or "task"


def _effective_run_name(req: GenerateRequest, task_id: str) -> str:
    base = _sanitize_run_name(req.run_name) if req.run_name else "task"
    return f"{base}__{task_id}"


def _get_task_or_404(task_id: str) -> TaskState:
    st = tasks.get(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="task not found")
    return st


def _resolve_curve_dir(st: TaskState, kind: str) -> Path:
    if not st.run_dir:
        raise HTTPException(status_code=409, detail="task output is not ready")
    if kind not in ("filtered", "rejected"):
        raise HTTPException(status_code=400, detail=f"unsupported curve kind: {kind}")
    dirname = "filtered_npy" if kind == "filtered" else "rejected_npy"
    curve_dir = Path(st.run_dir) / dirname
    if not curve_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"{dirname} not found: {curve_dir}")
    return curve_dir


def _load_manifest_map(st: TaskState, kind: str) -> Dict[str, dict]:
    if not st.run_dir:
        return {}
    if kind not in ("filtered", "rejected"):
        return {}
    manifest_name = "filtered_manifest.jsonl" if kind == "filtered" else "rejected_manifest.jsonl"
    manifest_path = Path(st.run_dir) / manifest_name
    if not manifest_path.is_file():
        return {}

    out: Dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            recon_path = rec.get("recon_path", "")
            if not recon_path:
                continue
            name = Path(recon_path).name
            out[name] = rec
    return out


def _extract_curve_metrics(rec: Optional[dict]) -> dict:
    if not rec:
        return {}
    keys = [
        "length_recon",
        "rg",
        "bond_mean",
        "bond_std",
        "bond_min",
        "bond_max",
        "bond_frac_out",
        "angle_mean",
        "angle_std",
        "angle_min",
        "angle_max",
        "angle_frac_out",
        "beta_total",
        "beta_max_run",
        "beta_in_sheet",
        "beta_sheet_fraction",
        "beta_strands_total",
        "beta_strands_sheet",
        "beta_strands_isolated",
        "n_self_clash_pairs",
        "n_seg_clash_pairs",
        "reject_reason",
    ]
    return {k: rec[k] for k in keys if k in rec}


def _load_curve_payload(path: Path, rec: Optional[dict] = None) -> dict:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise HTTPException(status_code=500, detail=f"bad curve shape in {path.name}: {tuple(arr.shape)}")

    xyz = arr[:, :3].astype(np.float32)
    ss = arr[:, 3:6].astype(np.float32)
    ss_idx = np.argmax(ss, axis=-1).astype(np.int32)

    return {
        "name": path.name,
        "length": int(arr.shape[0]),
        "xyz": xyz.tolist(),
        "ss_idx": ss_idx.tolist(),
        "ss_one_hot": ss.tolist(),
        "metrics": _extract_curve_metrics(rec),
    }


def _build_cmd(req: GenerateRequest, task_id: str) -> list[str]:
    run_name = _effective_run_name(req, task_id)
    return [
        "python",
        str(SCRIPT_PATH),
        "--ae_config", req.ae_config,
        "--ae_ckpt", req.ae_ckpt,
        "--features_pt", req.features_pt,
        "--ot_h", req.ot_h,
        "--out_root", req.out_root,
        "--run_name", run_name,
        "--n_generate", str(req.n_generate),
        "--num_gen_x", str(req.num_gen_x),
        "--ot_bat_size_n", str(req.ot_bat_size_n),
        "--ot_thresh", str(req.ot_thresh),
        "--decode_batch_size", str(req.decode_batch_size),
        "--min_length", "2",
        "--min_pairwise_dist", "2.0",
        "--neighbor_exclude", "2",
        "--ot_root", req.ot_root,
        "--gpu_id", str(req.gpu_id),
        "--select_random",
        "--seed", str(req.seed),
    ]


def _worker_loop() -> None:
    while True:
        task_id, req = task_queue.get()
        st = tasks[task_id]
        st.status = "running"
        st.started_at = time.time()

        cmd = _build_cmd(req, task_id)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(req.gpu_id)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            out = proc.stdout[-12000:] if proc.stdout else ""
            st.stdout_tail = out

            # best-effort parse run_dir from output
            run_dir = ""
            for line in out.splitlines()[::-1]:
                if line.strip().startswith("[done] outputs:"):
                    run_dir = line.split(":", 1)[1].strip()
                    break
            if not run_dir:
                run_name = _effective_run_name(req, task_id)
                run_dir = str(Path(req.out_root) / run_name)

            st.run_dir = run_dir
            st.summary_path = str(Path(run_dir) / "summary.json")

            if proc.returncode == 0:
                st.status = "done"
            else:
                st.status = "failed"
                st.error = f"generator exited with code {proc.returncode}"

        except Exception as e:
            st.status = "failed"
            st.error = str(e)
        finally:
            st.ended_at = time.time()
            task_queue.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "queue_size": task_queue.qsize(),
        "tasks": len(tasks),
        "single_gpu": True,
        "repo_root": str(REPO_ROOT),
    }


@app.get("/")
def root() -> RedirectResponse:
    if WEB_DIR.is_dir():
        return RedirectResponse(url="/web/")
    return RedirectResponse(url="/health")


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    if not Path(req.ae_ckpt).is_file():
        raise HTTPException(status_code=400, detail=f"ae_ckpt not found: {req.ae_ckpt}")
    if not Path(req.features_pt).is_file():
        raise HTTPException(status_code=400, detail=f"features_pt not found: {req.features_pt}")
    if not Path(req.ot_h).is_file():
        raise HTTPException(status_code=400, detail=f"ot_h not found: {req.ot_h}")
    if not Path(req.ot_root).is_dir():
        raise HTTPException(status_code=400, detail=f"ot_root not found: {req.ot_root}")

    task_id = uuid.uuid4().hex[:12]
    tasks[task_id] = TaskState(task_id=task_id)
    task_queue.put((task_id, req))
    effective_run_name = _effective_run_name(req, task_id)
    return {
        "task_id": task_id,
        "run_name": effective_run_name,
        "status": "queued",
        "queue_size": task_queue.qsize(),
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    st = _get_task_or_404(task_id)

    payload = {
        "task_id": st.task_id,
        "status": st.status,
        "created_at": st.created_at,
        "started_at": st.started_at,
        "ended_at": st.ended_at,
        "run_dir": st.run_dir,
        "summary_path": st.summary_path,
        "error": st.error,
    }

    if st.summary_path and Path(st.summary_path).is_file():
        try:
            with open(st.summary_path, "r", encoding="utf-8") as f:
                payload["summary"] = json.load(f)
        except Exception:
            payload["summary"] = None
    else:
        payload["summary"] = None

    return payload


@app.get("/tasks/{task_id}/curves")
def list_curves(task_id: str, limit: int = 12) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"task is not done yet: {st.status}")

    filtered_dir = _resolve_curve_dir(st, "filtered")
    manifest_map = _load_manifest_map(st, "filtered")
    limit = max(1, min(int(limit), 48))
    files = sorted(filtered_dir.glob("*.npy"))[:limit]

    return {
        "task_id": task_id,
        "run_dir": st.run_dir,
        "count": len(files),
        "curves": [_load_curve_payload(path, manifest_map.get(path.name)) for path in files],
    }


@app.get("/tasks/{task_id}/curves/{curve_name}")
def get_curve(task_id: str, curve_name: str) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"task is not done yet: {st.status}")

    filtered_dir = _resolve_curve_dir(st, "filtered")
    manifest_map = _load_manifest_map(st, "filtered")
    path = (filtered_dir / curve_name).resolve()
    if path.parent != filtered_dir.resolve() or path.suffix != ".npy" or not path.is_file():
        raise HTTPException(status_code=404, detail="curve not found")

    return _load_curve_payload(path, manifest_map.get(path.name))


@app.get("/tasks/{task_id}/rejected-curves")
def list_rejected_curves(task_id: str, limit: int = 24) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"task is not done yet: {st.status}")

    rejected_dir = _resolve_curve_dir(st, "rejected")
    manifest_map = _load_manifest_map(st, "rejected")
    limit = max(1, min(int(limit), 96))
    files = sorted(rejected_dir.glob("*.npy"))[:limit]

    return {
        "task_id": task_id,
        "run_dir": st.run_dir,
        "count": len(files),
        "curves": [_load_curve_payload(path, manifest_map.get(path.name)) for path in files],
    }


@app.get("/tasks/{task_id}/rejected-curves/{curve_name}")
def get_rejected_curve(task_id: str, curve_name: str) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"task is not done yet: {st.status}")

    rejected_dir = _resolve_curve_dir(st, "rejected")
    manifest_map = _load_manifest_map(st, "rejected")
    path = (rejected_dir / curve_name).resolve()
    if path.parent != rejected_dir.resolve() or path.suffix != ".npy" or not path.is_file():
        raise HTTPException(status_code=404, detail="curve not found")

    return _load_curve_payload(path, manifest_map.get(path.name))
