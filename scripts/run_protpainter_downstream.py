#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


PROTPAINTER_ROOT = Path("/data/zky/ProtPainter")


def _load_selection(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    curves = payload.get("curves", [])
    if not curves:
        raise RuntimeError(f"no selected curves found in {path}")
    return payload


def _ensure_import_paths() -> None:
    sys.path.insert(0, str(PROTPAINTER_ROOT))
    sys.path.insert(0, str(PROTPAINTER_ROOT / "C2S"))
    sys.path.insert(0, str(PROTPAINTER_ROOT / "LMPNN"))
    sys.path.insert(0, str(PROTPAINTER_ROOT / "OF"))
    sys.path.insert(0, str(PROTPAINTER_ROOT / "PFF"))


def _ss_labels_from_curve_array(arr: np.ndarray) -> list[str]:
    raw_ss = arr[:, 3:6]
    label_map = {0: "h", 1: "s", 2: "l"}
    return [label_map[int(idx)] for idx in np.argmax(raw_ss, axis=1)]


def _convert_selected_curves(selection: dict, curves_dir: Path) -> list[dict]:
    from utils import write_pdb

    converted = []
    curves_dir.mkdir(parents=True, exist_ok=True)
    for item in selection["curves"]:
        source = Path(item["curve_path"])
        arr = np.load(source, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise RuntimeError(f"unexpected selected curve shape for {source}: {tuple(arr.shape)}")
        curve_coords = arr[:, :3].astype(np.float32)
        ss_labels = _ss_labels_from_curve_array(arr)
        out_name = f"{source.stem}_curve.npy"
        out_path = curves_dir / out_name
        payload = {
            "curve_coords": curve_coords,
            "ss_labels": ss_labels,
            "original_pdb": str(source),
            "num_curve_points": int(curve_coords.shape[0]),
        }
        np.save(out_path, payload, allow_pickle=True)
        write_pdb(curve_coords - np.mean(curve_coords, axis=0, keepdims=True), out_path.with_suffix(".pdb"))
        converted.append({
            "name": out_name,
            "source_curve": str(source),
            "converted_curve": str(out_path),
            "preview_pdb": str(out_path.with_suffix(".pdb")),
        })
    return converted


def _load_c2s_model(device: torch.device):
    import C2S.c2s_config as c2s_config
    from C2S.c2s_model import make_model

    c2s_config.model_path = str(PROTPAINTER_ROOT / "C2S" / "experiment" / "model_best_rots.pth")
    c2s_config.device = device
    model = make_model(
        c2s_config.src_vocab_size,
        c2s_config.tgt_vocab_size,
        c2s_config.n_layers,
        c2s_config.d_model,
        c2s_config.d_ff,
        c2s_config.n_heads,
        c2s_config.dropout,
    )
    state = torch.load(c2s_config.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_curve_to_sketch(curves_dir: Path, sketch_dir: Path, device: torch.device) -> list[str]:
    from C2S.c2s_main import curve2sketch

    sketch_dir.mkdir(parents=True, exist_ok=True)
    model = _load_c2s_model(device)
    curve2sketch(str(curves_dir), str(sketch_dir), model)
    return [str(path) for path in sorted(sketch_dir.glob("*_sketch.npy"))]


def _load_pff_model(device: torch.device):
    from omegaconf import OmegaConf
    from PFF.models.flow_model import FlowModel

    inference_cfg = OmegaConf.load(PROTPAINTER_ROOT / "PFF" / "configs" / "inference_unconditional.yaml")
    ckpt_cfg = OmegaConf.load(PROTPAINTER_ROOT / "PFF" / "weights" / "pdb" / "config.yaml")
    OmegaConf.set_struct(inference_cfg, False)
    OmegaConf.set_struct(ckpt_cfg, False)
    cfg = OmegaConf.merge(inference_cfg, ckpt_cfg)
    cfg.experiment.checkpointer.dirpath = str(PROTPAINTER_ROOT / "PFF")
    cfg.model.ipa.c_hidden = 128
    cfg.model.edge_features.embed_diffuse_mask = False

    model = FlowModel(cfg.model)
    ckpt_path = PROTPAINTER_ROOT / "PFF" / "weights" / "pdb" / "published_main.ckpt"
    model = model.to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = state["state_dict"]
    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_state_dict[key.replace("model.", "")] = value
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


def _run_sketch_to_backbone(sketch_paths: list[str], backbone_dir: Path, device: torch.device, num_bbs: int) -> list[dict]:
    import conditional_gen
    from conditional_gen import run as pff_run

    original_dssp = conditional_gen.calculate_dssp_percentages

    def _safe_dssp(*args, **kwargs):
        result = original_dssp(*args, **kwargs)
        if result is None:
            return {
                "helix_count": 0,
                "strand_count": 0,
                "total_residues": 1,
                "helix_pct": 0.0,
                "strand_pct": 0.0,
            }
        return result

    conditional_gen.calculate_dssp_percentages = _safe_dssp

    backbone_dir.mkdir(parents=True, exist_ok=True)
    model = _load_pff_model(device)
    outputs = []
    for sketch_path in sketch_paths:
        sketch_info = np.load(sketch_path, allow_pickle=True).item()
        coords = np.array(sketch_info["predicted_s"], dtype=np.float32)[:, :3]
        coords_s = np.array(sketch_info["coords_c"], dtype=np.float32)
        gt_markers = coords_s[:, 3:8]
        total_residues = max(len(gt_markers), 1)
        h_pct = float(np.sum(gt_markers[:, 3]) / total_residues)
        s_pct = float(np.sum(gt_markers[:, 4]) / total_residues)

        item_root = backbone_dir / Path(sketch_path).stem.replace("_sketch", "")
        pff_run(
            str(item_root),
            coords,
            rots=None,
            model=model,
            h_pct=h_pct,
            s_pct=s_pct,
            alpha=1.0,
            beta=10.0,
            gamma=0.2,
            mode="mainly_alpha",
            total_num=num_bbs,
        )
        sample_pdbs = [str(path) for path in sorted((item_root / "gen_pdbs").glob("*/sample.pdb"))]
        outputs.append({
            "name": item_root.name,
            "root_dir": str(item_root),
            "sketch_npy": str(sketch_path),
            "sketch_pdb": str(item_root / "mysketch.pdb"),
            "sample_pdbs": sample_pdbs,
        })
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_bbs", type=int, default=1)
    args = parser.parse_args()

    selection_manifest = Path(args.selection_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "downstream_summary.json"

    try:
        _ensure_import_paths()
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        selection = _load_selection(selection_manifest)
        os.chdir(output_root)

        curves_dir = output_root / "curves"
        sketch_dir = output_root / "sketches"
        backbone_dir = output_root / "backbone"

        converted = _convert_selected_curves(selection, curves_dir)
        sketch_paths = _run_curve_to_sketch(curves_dir, sketch_dir, device)
        backbone_outputs = _run_sketch_to_backbone(sketch_paths, backbone_dir, device, args.num_bbs)

        summary = {
            "status": "done",
            "selection_manifest": str(selection_manifest),
            "output_root": str(output_root),
            "selected_count": len(selection.get("curves", [])),
            "converted_curves": converted,
            "sketches": sketch_paths,
            "backbones": backbone_outputs,
            "stages": {
                "curve_selection": {"status": "done", "count": len(selection.get("curves", []))},
                "sketch": {"status": "done", "count": len(sketch_paths)},
                "backbone": {"status": "done", "count": len(backbone_outputs)},
                "sequence": {"status": "planned"},
                "folded": {"status": "planned"},
                "evaluation": {"status": "planned"},
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[done] downstream outputs: {output_root}")
        return 0
    except Exception as e:
        summary = {
            "status": "failed",
            "selection_manifest": str(selection_manifest),
            "output_root": str(output_root),
            "error": str(e),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
