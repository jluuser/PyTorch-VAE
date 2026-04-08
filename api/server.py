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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


REPO_ROOT = Path("/home/zky/PyTorch-VAE")
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_aeot_end2end.py"
DOWNSTREAM_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_protpainter_downstream.py"
DEFAULT_AE_CONFIG = str(REPO_ROOT / "configs" / "stage1_ae.yaml")
DEFAULT_AE_CKPT = "/home/zky/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt"
DEFAULT_FEATURES_PT = "/home/zky/AE-OT/results_curves/features_5w.pt"
DEFAULT_OT_H = "/home/zky/AE-OT/results_curves/h.pt"
DEFAULT_OT_ROOT = "/home/zky/AE-OT"
DEFAULT_OUT_ROOT = "/data/zky/api_results"
DEFAULT_GPU_ID = 0
DEFAULT_DOWNSTREAM_PYTHON = os.environ.get("PROTPAINTER_PYTHON", "python")

HELIX_CONSTRAINTS = {
    "a": (89.0, 12.0),
    "d": (50.0, 20.0),
    "d2": (5.5, 0.5),
    "d3": (5.3, 0.5),
    "d4": (6.4, 0.6),
}
STRAND_CONSTRAINTS = {
    "a": (124.0, 14.0),
    "d": (-170.0, 45.0),
    "d2": (6.7, 0.6),
    "d3": (9.9, 0.9),
    "d4": (12.4, 1.1),
}
HELIX_SIZE = 5
STRAND_SIZE = 4


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


class DownstreamSelectionRequest(BaseModel):
    curve_names: list[str] = Field(default_factory=list, description="Filtered curve filenames selected for downstream pipeline")


@dataclass
class TaskState:
    task_id: str
    gpu_id: int = DEFAULT_GPU_ID
    status: str = "queued"  # queued/running/done/failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    run_dir: str = ""
    summary_path: str = ""
    stdout_tail: str = ""
    error: str = ""
    downstream_status: str = "idle"  # idle/queued/running/done/failed
    downstream_started_at: Optional[float] = None
    downstream_ended_at: Optional[float] = None
    downstream_dir: str = ""
    downstream_summary_path: str = ""
    downstream_error: str = ""
    downstream_stdout_tail: str = ""
    downstream_action: str = "backbone"  # backbone / sequence_fold / evaluation


app = FastAPI(title="AEOT Single-GPU API", version="1.0.0")
WEB_DIR = REPO_ROOT / "api" / "web"
if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

task_queue: "queue.Queue[tuple[str, GenerateRequest]]" = queue.Queue()
downstream_queue: "queue.Queue[str]" = queue.Queue()
tasks: Dict[str, TaskState] = {}
gpu_job_lock = threading.Lock()


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


def _np_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def _np_angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    v1 = x - y
    v2 = z - y
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    denom = np.maximum(denom, 1e-8)
    cosv = np.sum(v1 * v2, axis=-1) / denom
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def _np_dihedral(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    b0 = w - x
    b1 = y - x
    b2 = z - y
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1 = b1 / np.maximum(b1_norm, 1e-8)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    wv = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x_ = np.sum(v * wv, axis=-1)
    y_ = np.sum(np.cross(b1, v) * wv, axis=-1)
    return np.degrees(np.arctan2(y_, x_))


def _cond_to_pred_np(cond: np.ndarray, size: int) -> np.ndarray:
    n = cond.shape[0]
    if n <= 0:
        return np.zeros(0, dtype=bool)
    if n < size:
        return np.zeros(n, dtype=bool)
    window_ok = np.array([bool(np.all(cond[i:i + size])) for i in range(n - size + 1)], dtype=bool)
    pred = np.zeros(n, dtype=bool)
    for i, ok in enumerate(window_ok):
        if ok:
            pred[i:i + size] = True
    return pred


def _assign_ss_idx_from_xyz(xyz: np.ndarray) -> np.ndarray:
    n = int(xyz.shape[0])
    if n < 5:
        return np.full(n, 2, dtype=np.int32)

    x0 = xyz[:-4]
    x1 = xyz[1:-3]
    x2 = xyz[2:-2]
    x3 = xyz[3:-1]
    x4 = xyz[4:]

    values = {
        "a": _np_angle(x0, x1, x2),
        "d": _np_dihedral(x0, x1, x2, x3),
        "d2": _np_distance(x2, x0),
        "d3": _np_distance(x3, x0),
        "d4": _np_distance(x4, x0),
    }

    helix_cond = {}
    for key, (center, tol) in HELIX_CONSTRAINTS.items():
        helix_cond[key] = (values[key] >= center - tol) & (values[key] <= center + tol)
    strand_cond = {}
    for key, (center, tol) in STRAND_CONSTRAINTS.items():
        strand_cond[key] = (values[key] >= center - tol) & (values[key] <= center + tol)

    cond_helix = (helix_cond["d3"] & helix_cond["d4"]) | (helix_cond["a"] & helix_cond["d"])
    cond_strand = ((strand_cond["d2"] & strand_cond["d3"] & strand_cond["d4"]) | (strand_cond["a"] & strand_cond["d"]))

    is_helix_core = _cond_to_pred_np(cond_helix, HELIX_SIZE)
    is_strand_core = _cond_to_pred_np(cond_strand, STRAND_SIZE)

    is_helix = np.pad(is_helix_core, (1, 3), constant_values=False)[:n]
    is_strand = np.pad(is_strand_core, (1, 3), constant_values=False)[:n]
    is_strand = is_strand & (~is_helix)

    ss_idx = np.full(n, 2, dtype=np.int32)
    ss_idx[is_strand] = 1
    ss_idx[is_helix] = 0
    return ss_idx


def _load_pdb_trace(path: Path) -> dict:
    xyz = []
    atom_names = []
    residue_ids = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            xyz.append([x, y, z])
            atom_names.append(atom_name)
            residue_ids.append(line[22:26].strip())
    if not xyz:
        raise HTTPException(status_code=500, detail=f"no CA trace found in pdb: {path.name}")
    xyz_arr = np.asarray(xyz, dtype=np.float32)
    ss_idx = _assign_ss_idx_from_xyz(xyz_arr)
    return {
        "name": path.name,
        "length": len(xyz_arr),
        "xyz": xyz_arr.tolist(),
        "ss_idx": ss_idx.tolist(),
        "atom_name": atom_names,
        "residue_id": residue_ids,
    }


def _kabsch_align(ref_xyz: np.ndarray, mobile_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref_xyz), len(mobile_xyz))
    if n <= 0:
        raise HTTPException(status_code=400, detail="no overlapping trace points for alignment")
    ref = np.asarray(ref_xyz[:n], dtype=np.float64)
    mob = np.asarray(mobile_xyz[:n], dtype=np.float64)
    ref_centroid = ref.mean(axis=0)
    mob_centroid = mob.mean(axis=0)
    ref_c = ref - ref_centroid
    mob_c = mob - mob_centroid
    h = mob_c.T @ ref_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    mob_aligned = (mob_c @ r.T) + ref_centroid
    return ref.astype(np.float32), mob_aligned.astype(np.float32)


def _kabsch_transform(ref_xyz: np.ndarray, mobile_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = min(len(ref_xyz), len(mobile_xyz))
    if n <= 0:
        raise HTTPException(status_code=400, detail="no overlapping trace points for alignment")
    ref = np.asarray(ref_xyz[:n], dtype=np.float64)
    mob = np.asarray(mobile_xyz[:n], dtype=np.float64)
    ref_centroid = ref.mean(axis=0)
    mob_centroid = mob.mean(axis=0)
    ref_c = ref - ref_centroid
    mob_c = mob - mob_centroid
    h = mob_c.T @ ref_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return ref_centroid.astype(np.float64), mob_centroid.astype(np.float64), r.astype(np.float64), n


def _transform_pdb_text(path: Path, ref_centroid: np.ndarray, mobile_centroid: np.ndarray, rotation: np.ndarray) -> str:
    out_lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    out_lines.append(line)
                    continue
                xyz = np.asarray([x, y, z], dtype=np.float64)
                new_xyz = ((xyz - mobile_centroid) @ rotation.T) + ref_centroid
                line = (
                    f"{line[:30]}"
                    f"{new_xyz[0]:8.3f}{new_xyz[1]:8.3f}{new_xyz[2]:8.3f}"
                    f"{line[54:]}"
                )
            out_lines.append(line)
    return "".join(out_lines)


def _selection_dir(st: TaskState) -> Path:
    if not st.run_dir:
        raise HTTPException(status_code=409, detail="task output is not ready")
    return Path(st.run_dir) / "selected_curves"


def _selection_path(st: TaskState) -> Path:
    return _selection_dir(st) / "selected_manifest.json"


def _default_downstream_stages() -> list[dict]:
    return [
        {"key": "sketch", "label": "Sketch", "status": "planned"},
        {"key": "backbone", "label": "Backbone", "status": "planned"},
        {"key": "sequence", "label": "Sequence", "status": "planned"},
        {"key": "folded", "label": "Folded", "status": "planned"},
        {"key": "evaluation", "label": "Evaluation", "status": "planned"},
    ]


def _empty_downstream_selection(st: TaskState) -> dict:
    return {
        "task_id": st.task_id,
        "run_dir": st.run_dir,
        "selection_path": str(_selection_path(st)),
        "selected_count": 0,
        "selected_at": None,
        "curves": [],
        "stages": _default_downstream_stages(),
    }


def _load_downstream_selection(st: TaskState) -> dict:
    manifest_path = _selection_path(st)
    if not manifest_path.is_file():
        return _empty_downstream_selection(st)
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read downstream selection: {e}")
    payload.setdefault("task_id", st.task_id)
    payload.setdefault("run_dir", st.run_dir)
    payload.setdefault("selection_path", str(manifest_path))
    payload.setdefault("selected_count", len(payload.get("curves", [])))
    payload.setdefault("stages", _default_downstream_stages())
    return payload


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


def _downstream_output_dir(st: TaskState) -> Path:
    if not st.run_dir:
        raise HTTPException(status_code=409, detail="task output is not ready")
    return Path(st.run_dir) / "downstream"


def _build_downstream_cmd(st: TaskState) -> list[str]:
    selection_path = _selection_path(st)
    return [
        DEFAULT_DOWNSTREAM_PYTHON,
        str(DOWNSTREAM_SCRIPT_PATH),
        "--selection_manifest", str(selection_path),
        "--output_root", str(_downstream_output_dir(st)),
        "--gpu_id", str(st.gpu_id),
        "--num_bbs", "1",
        "--stage", st.downstream_action,
        "--num_seqs", "4",
        "--lmpnn_temperature", "0.1",
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
            with gpu_job_lock:
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


def _downstream_worker_loop() -> None:
    while True:
        task_id = downstream_queue.get()
        st = tasks[task_id]
        st.downstream_status = "running"
        st.downstream_started_at = time.time()
        st.downstream_error = ""

        cmd = _build_downstream_cmd(st)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(st.gpu_id)

        try:
            with gpu_job_lock:
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
            st.downstream_stdout_tail = out
            downstream_dir = str(_downstream_output_dir(st))
            st.downstream_dir = downstream_dir
            st.downstream_summary_path = str(Path(downstream_dir) / "downstream_summary.json")
            if proc.returncode == 0:
                st.downstream_status = "done"
            else:
                st.downstream_status = "failed"
                st.downstream_error = f"downstream exited with code {proc.returncode}"
        except Exception as e:
            st.downstream_status = "failed"
            st.downstream_error = str(e)
        finally:
            st.downstream_ended_at = time.time()
            downstream_queue.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()
_downstream_worker_thread = threading.Thread(target=_downstream_worker_loop, daemon=True)
_downstream_worker_thread.start()


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/web"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "queue_size": task_queue.qsize(),
        "downstream_queue_size": downstream_queue.qsize(),
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
    tasks[task_id] = TaskState(task_id=task_id, gpu_id=req.gpu_id)
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


@app.get("/tasks/{task_id}/downstream-selection")
def get_downstream_selection(task_id: str) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"task is not done yet: {st.status}")
    return _load_downstream_selection(st)


@app.post("/tasks/{task_id}/downstream-selection")
def save_downstream_selection(task_id: str, req: DownstreamSelectionRequest) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"task is not done yet: {st.status}")

    filtered_dir = _resolve_curve_dir(st, "filtered")
    manifest_map = _load_manifest_map(st, "filtered")

    normalized_names: list[str] = []
    seen: set[str] = set()
    for raw_name in req.curve_names:
        name = Path(str(raw_name)).name
        if name in seen:
            continue
        curve_path = (filtered_dir / name).resolve()
        if curve_path.parent != filtered_dir.resolve() or curve_path.suffix != ".npy" or not curve_path.is_file():
            raise HTTPException(status_code=400, detail=f"filtered curve not found: {name}")
        normalized_names.append(name)
        seen.add(name)

    records = []
    for name in normalized_names:
        rec = manifest_map.get(name)
        curve_path = filtered_dir / name
        curve_payload = _load_curve_payload(curve_path, rec)
        records.append({
            "name": name,
            "curve_path": str(curve_path),
            "length": curve_payload["length"],
            "metrics": curve_payload["metrics"],
            "downstream_status": "planned",
        })

    selection_dir = _selection_dir(st)
    selection_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _selection_path(st)
    payload = {
        "task_id": st.task_id,
        "run_dir": st.run_dir,
        "selection_path": str(manifest_path),
        "selected_count": len(records),
        "selected_at": time.time(),
        "curves": records,
        "stages": _default_downstream_stages(),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


@app.get("/tasks/{task_id}/downstream")
def get_downstream_status(task_id: str) -> dict:
    st = _get_task_or_404(task_id)
    payload = {
        "task_id": st.task_id,
        "action": st.downstream_action,
        "status": st.downstream_status,
        "started_at": st.downstream_started_at,
        "ended_at": st.downstream_ended_at,
        "output_dir": st.downstream_dir,
        "summary_path": st.downstream_summary_path,
        "error": st.downstream_error,
    }
    summary_path = Path(st.downstream_summary_path) if st.downstream_summary_path else None
    if summary_path and summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                payload["summary"] = json.load(f)
        except Exception:
            payload["summary"] = None
    else:
        payload["summary"] = None
    return payload


@app.post("/tasks/{task_id}/run-downstream")
def run_downstream(task_id: str) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"curve task is not done yet: {st.status}")
    selection = _load_downstream_selection(st)
    if selection.get("selected_count", 0) <= 0:
        raise HTTPException(status_code=400, detail="no selected curves found for downstream pipeline")
    if st.downstream_status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"downstream job is already {st.downstream_status}")

    st.downstream_action = "backbone"
    st.downstream_status = "queued"
    st.downstream_started_at = None
    st.downstream_ended_at = None
    st.downstream_error = ""
    st.downstream_stdout_tail = ""
    st.downstream_dir = str(_downstream_output_dir(st))
    st.downstream_summary_path = str(Path(st.downstream_dir) / "downstream_summary.json")
    downstream_queue.put(task_id)
    return {
        "task_id": task_id,
        "action": st.downstream_action,
        "status": st.downstream_status,
        "queue_size": downstream_queue.qsize(),
        "selection_path": selection.get("selection_path"),
        "selected_count": selection.get("selected_count", 0),
        "output_dir": st.downstream_dir,
    }


@app.post("/tasks/{task_id}/run-sequence-fold")
def run_sequence_fold(task_id: str) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"curve task is not done yet: {st.status}")
    if st.downstream_status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"downstream job is already {st.downstream_status}")
    summary_path = Path(st.downstream_summary_path) if st.downstream_summary_path else _downstream_output_dir(st) / "downstream_summary.json"
    if not summary_path.is_file():
        raise HTTPException(status_code=409, detail="backbone downstream summary not found; run Sketch / Backbone first")
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read downstream summary: {e}")
    if not summary.get("backbones"):
        raise HTTPException(status_code=409, detail="no backbone outputs found; run Sketch / Backbone first")

    st.downstream_action = "sequence_fold"
    st.downstream_status = "queued"
    st.downstream_started_at = None
    st.downstream_ended_at = None
    st.downstream_error = ""
    st.downstream_stdout_tail = ""
    st.downstream_dir = str(_downstream_output_dir(st))
    st.downstream_summary_path = str(Path(st.downstream_dir) / "downstream_summary.json")
    downstream_queue.put(task_id)
    return {
        "task_id": task_id,
        "action": st.downstream_action,
        "status": st.downstream_status,
        "queue_size": downstream_queue.qsize(),
        "output_dir": st.downstream_dir,
    }


@app.post("/tasks/{task_id}/run-evaluation")
def run_evaluation(task_id: str) -> dict:
    st = _get_task_or_404(task_id)
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"curve task is not done yet: {st.status}")
    if st.downstream_status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"downstream job is already {st.downstream_status}")
    summary_path = Path(st.downstream_summary_path) if st.downstream_summary_path else _downstream_output_dir(st) / "downstream_summary.json"
    if not summary_path.is_file():
        raise HTTPException(status_code=409, detail="downstream summary not found; run previous stages first")
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read downstream summary: {e}")
    if not summary.get("sequence_outputs"):
        raise HTTPException(status_code=409, detail="no sequence/folded outputs found; run Sequence / Folded first")

    st.downstream_action = "evaluation"
    st.downstream_status = "queued"
    st.downstream_started_at = None
    st.downstream_ended_at = None
    st.downstream_error = ""
    st.downstream_stdout_tail = ""
    st.downstream_dir = str(_downstream_output_dir(st))
    st.downstream_summary_path = str(Path(st.downstream_dir) / "downstream_summary.json")
    downstream_queue.put(task_id)
    return {
        "task_id": task_id,
        "action": st.downstream_action,
        "status": st.downstream_status,
        "queue_size": downstream_queue.qsize(),
        "output_dir": st.downstream_dir,
    }


@app.get("/tasks/{task_id}/downstream-pdb")
def get_downstream_pdb(task_id: str, path: str) -> dict:
    st = _get_task_or_404(task_id)
    if not st.downstream_dir:
        raise HTTPException(status_code=409, detail="downstream output is not ready")

    base_dir = Path(st.downstream_dir).resolve()
    pdb_path = Path(path).resolve()
    if pdb_path.suffix.lower() != ".pdb" or not pdb_path.is_file() or base_dir not in pdb_path.parents:
        raise HTTPException(status_code=404, detail="downstream pdb not found")
    payload = _load_pdb_trace(pdb_path)
    payload["path"] = str(pdb_path)
    return payload


@app.get("/tasks/{task_id}/downstream-pdb-text")
def get_downstream_pdb_text(task_id: str, path: str) -> PlainTextResponse:
    st = _get_task_or_404(task_id)
    if not st.downstream_dir:
        raise HTTPException(status_code=409, detail="downstream output is not ready")

    base_dir = Path(st.downstream_dir).resolve()
    pdb_path = Path(path).resolve()
    if pdb_path.suffix.lower() != ".pdb" or not pdb_path.is_file() or base_dir not in pdb_path.parents:
        raise HTTPException(status_code=404, detail="downstream pdb not found")
    try:
        text = pdb_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read pdb: {e}")
    return PlainTextResponse(text)


@app.get("/tasks/{task_id}/downstream-compare")
def get_downstream_compare(task_id: str, ref: str, mobile: str) -> dict:
    st = _get_task_or_404(task_id)
    if not st.downstream_dir:
        raise HTTPException(status_code=409, detail="downstream output is not ready")

    base_dir = Path(st.downstream_dir).resolve()
    ref_path = Path(ref).resolve()
    mobile_path = Path(mobile).resolve()
    for path in (ref_path, mobile_path):
        if path.suffix.lower() != ".pdb" or not path.is_file() or base_dir not in path.parents:
            raise HTTPException(status_code=404, detail=f"compare pdb not found: {path}")

    ref_payload = _load_pdb_trace(ref_path)
    mobile_payload = _load_pdb_trace(mobile_path)
    ref_xyz, mobile_xyz = _kabsch_align(
        np.asarray(ref_payload["xyz"], dtype=np.float32),
        np.asarray(mobile_payload["xyz"], dtype=np.float32),
    )
    n = min(len(ref_xyz), len(mobile_xyz), len(ref_payload["ss_idx"]), len(mobile_payload["ss_idx"]))
    return {
        "ref_path": str(ref_path),
        "mobile_path": str(mobile_path),
        "ref_name": ref_path.name,
        "mobile_name": mobile_path.name,
        "length": int(n),
        "ref_xyz": ref_xyz[:n].tolist(),
        "mobile_xyz": mobile_xyz[:n].tolist(),
        "ref_ss_idx": ref_payload["ss_idx"][:n],
        "mobile_ss_idx": mobile_payload["ss_idx"][:n],
    }


@app.get("/tasks/{task_id}/downstream-compare-pdb-text")
def get_downstream_compare_pdb_text(task_id: str, ref: str, mobile: str) -> dict:
    st = _get_task_or_404(task_id)
    if not st.downstream_dir:
        raise HTTPException(status_code=409, detail="downstream output is not ready")

    base_dir = Path(st.downstream_dir).resolve()
    ref_path = Path(ref).resolve()
    mobile_path = Path(mobile).resolve()
    for path in (ref_path, mobile_path):
        if path.suffix.lower() != ".pdb" or not path.is_file() or base_dir not in path.parents:
            raise HTTPException(status_code=404, detail=f"compare pdb not found: {path}")

    ref_payload = _load_pdb_trace(ref_path)
    mobile_payload = _load_pdb_trace(mobile_path)
    ref_centroid, mobile_centroid, rotation, n = _kabsch_transform(
        np.asarray(ref_payload["xyz"], dtype=np.float32),
        np.asarray(mobile_payload["xyz"], dtype=np.float32),
    )
    try:
        ref_text = ref_path.read_text(encoding="utf-8", errors="ignore")
        mobile_text = _transform_pdb_text(mobile_path, ref_centroid, mobile_centroid, rotation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to build aligned pdb text: {e}")

    return {
        "ref_path": str(ref_path),
        "mobile_path": str(mobile_path),
        "ref_name": ref_path.name,
        "mobile_name": mobile_path.name,
        "length": int(n),
        "ref_pdb": ref_text,
        "mobile_pdb": mobile_text,
    }
