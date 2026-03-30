#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


PROTPAINTER_ROOT = Path("/data/zky/ProtPainter")
LMPNN_MODEL_DIR = PROTPAINTER_ROOT / "LMPNN" / "model_params"
OF_WEIGHTS_PATH = PROTPAINTER_ROOT / "OF" / "release1.pt"
USALIGN_PATH = PROTPAINTER_ROOT / "USalign" / "USalign"


def _load_selection(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    curves = payload.get("curves", [])
    if not curves:
        raise RuntimeError(f"no selected curves found in {path}")
    return payload


def _load_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_summary(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def _parse_fasta(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    entries: list[dict[str, str]] = []
    header = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append({"header": header, "sequence": "".join(seq_parts)})
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        entries.append({"header": header, "sequence": "".join(seq_parts)})
    return entries


def _build_omegafold_args(device: str):
    import argparse as _argparse
    from OF.omegafold import pipeline as of_pipeline

    state_dict = of_pipeline._load_weights(
        "https://helixon.s3.amazonaws.com/release1.pt",
        str(OF_WEIGHTS_PATH),
    )
    state_dict = state_dict.pop("model", state_dict)
    args = _argparse.Namespace(
        input_file=None,
        output_dir=None,
        num_cycle=10,
        subbatch_size=None,
        device=device,
        weights_file=str(OF_WEIGHTS_PATH),
        weights="https://helixon.s3.amazonaws.com/release1.pt",
        model=1,
        pseudo_msa_mask_rate=0.12,
        num_pseudo_msa=15,
        allow_tf32=True,
    )
    forward_config = _argparse.Namespace(
        subbatch_size=args.subbatch_size,
        num_recycle=args.num_cycle,
    )
    return args, state_dict, forward_config


def _run_usalign(ref_pdb: str, mobile_pdb: str) -> dict[str, float]:
    proc = subprocess.run(
        [str(USALIGN_PATH), ref_pdb, mobile_pdb, "-TMscore", "1", "-outfmt", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or len(lines) < 2:
        raise RuntimeError(f"USalign failed for {Path(ref_pdb).name} vs {Path(mobile_pdb).name}: {proc.stdout[-400:]}")
    parts = lines[1].split("\t")
    if len(parts) < 5:
        raise RuntimeError(f"unexpected USalign output: {lines[1]}")
    tm1 = float(parts[2])
    tm2 = float(parts[3])
    rmsd = float(parts[4])
    return {
        "tm1": tm1,
        "tm2": tm2,
        "tm": (tm1 + tm2) / 2.0,
        "rmsd": rmsd,
    }


def _compute_ss_accuracy(sketch_pdb: str, backbone_pdb: str) -> float:
    from utils import assign_secondary_structures, extract_ca_coordinates

    sketch_ca = extract_ca_coordinates(sketch_pdb).unsqueeze(0)
    backbone_ca = extract_ca_coordinates(backbone_pdb).unsqueeze(0)
    sketch_ss = assign_secondary_structures(sketch_ca)[0]
    backbone_ss = assign_secondary_structures(backbone_ca)[0]
    if len(sketch_ss) != len(backbone_ss) or len(sketch_ss) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(sketch_ss, backbone_ss) if ((a == "h" and b == "h") or (a == "s" and b == "s") or a == "l"))
    return matches / len(sketch_ss)


def _run_sequence_and_fold(summary: dict, output_root: Path, device: torch.device, num_seqs: int, temperature: float) -> list[dict]:
    import conditional_gen
    import LMPNN.run as lmpnn_run

    backbone_items = summary.get("backbones", [])
    pdb_files: list[str] = []
    for item in backbone_items:
        pdb_files.extend(item.get("sample_pdbs", []))
    if not pdb_files:
        raise RuntimeError("no backbone sample.pdb files found; run sketch/backbone first")

    orig_lmpnn_main = conditional_gen.lmpnn_main
    orig_get_args = conditional_gen.pipeline.get_args
    orig_sys_argv = sys.argv[:]

    def _patched_lmpnn_main(model_type, seed, pdb_path_multi, temperature, num_seqs, device):
        orig_parse_args = lmpnn_run.argparse.ArgumentParser.parse_args

        def _patched_parse_args(self, args=None, namespace=None):
            ns = orig_parse_args(self, [], namespace)
            ns.checkpoint_protein_mpnn = str(LMPNN_MODEL_DIR / "proteinmpnn_v_48_020.pt")
            ns.checkpoint_ligand_mpnn = str(LMPNN_MODEL_DIR / "ligandmpnn_v_32_010_25.pt")
            ns.checkpoint_per_residue_label_membrane_mpnn = str(LMPNN_MODEL_DIR / "per_residue_label_membrane_mpnn_v_48_020.pt")
            ns.checkpoint_global_label_membrane_mpnn = str(LMPNN_MODEL_DIR / "global_label_membrane_mpnn_v_48_020.pt")
            ns.checkpoint_soluble_mpnn = str(LMPNN_MODEL_DIR / "solublempnn_v_48_020.pt")
            ns.checkpoint_path_sc = str(LMPNN_MODEL_DIR / "ligandmpnn_sc_v_32_002_16.pt")
            return ns

        lmpnn_run.argparse.ArgumentParser.parse_args = _patched_parse_args
        try:
            return orig_lmpnn_main(model_type, seed, pdb_path_multi, temperature, num_seqs, device)
        finally:
            lmpnn_run.argparse.ArgumentParser.parse_args = orig_parse_args

    conditional_gen.lmpnn_main = _patched_lmpnn_main
    conditional_gen.pipeline.get_args = lambda device=None: _build_omegafold_args(str(device))
    sys.argv = [sys.argv[0]]

    try:
        conditional_gen.myeval(
            pdb_files=pdb_files,
            curve_dir=str(output_root / "curves"),
            LMPNN_temperature=temperature,
            num_seqs=num_seqs,
            results_path=str(output_root / "evaluation" / "results.txt"),
            write_mode="w",
            device=str(device),
            cal_metrics=False,
        )
    finally:
        conditional_gen.lmpnn_main = orig_lmpnn_main
        conditional_gen.pipeline.get_args = orig_get_args
        sys.argv = orig_sys_argv

    outputs: list[dict[str, Any]] = []
    for item in backbone_items:
        item_root = Path(item["root_dir"])
        fasta_paths: list[str] = []
        folded_pdbs: list[str] = []
        sequences: list[dict[str, str]] = []

        for sample_pdb in item.get("sample_pdbs", []):
            sample_root = Path(sample_pdb).parent
            fasta_path = sample_root / "seqs" / "sample.fa"
            rec_dir = sample_root / "recs"
            if fasta_path.is_file():
                fasta_paths.append(str(fasta_path))
                sequences.extend(_parse_fasta(fasta_path))
            for path in sorted(rec_dir.glob("rec_*.pdb")):
                if path.stem.endswith("_curve"):
                    continue
                folded_pdbs.append(str(path))

        item["fasta_path"] = fasta_paths[0] if fasta_paths else ""
        item["fasta_paths"] = fasta_paths
        item["sequences"] = sequences
        item["folded_pdbs"] = folded_pdbs
        outputs.append(
            {
                "name": item["name"],
                "root_dir": str(item_root),
                "fasta_path": item["fasta_path"],
                "fasta_paths": fasta_paths,
                "sequence_count": len(sequences),
                "sequences": sequences,
                "folded_pdbs": folded_pdbs,
            }
        )
    return outputs


def _run_evaluation(summary: dict, output_root: Path) -> list[dict]:
    from utils import calculate_plddt, curve_similarity

    outputs: list[dict[str, Any]] = []
    sequence_outputs = summary.get("sequence_outputs", [])
    for item in sequence_outputs:
        name = item["name"]
        backbone_match = next((bb for bb in summary.get("backbones", []) if bb.get("name") == name), None)
        if not backbone_match:
            continue
        sample_pdbs = backbone_match.get("sample_pdbs", [])
        if not sample_pdbs:
            continue
        backbone_pdb = sample_pdbs[0]
        sketch_pdb = backbone_match.get("sketch_pdb", "")
        source_curve_path = output_root / "curves" / f"{name}_curve.npy"
        if not source_curve_path.is_file():
            raise RuntimeError(f"source curve not found for evaluation: {source_curve_path}")
        raw_curve = np.load(source_curve_path, allow_pickle=True).item()["curve_coords"]
        raw_curve_t = torch.tensor(raw_curve)

        folded_results: list[dict[str, Any]] = []
        for folded_pdb in item.get("folded_pdbs", []):
            folded_path = Path(folded_pdb)
            base_stem = folded_path.stem[:-6] if folded_path.stem.endswith("_curve") else folded_path.stem
            rec_curve_npy = folded_path.with_name(f"{base_stem}_curve.npy")
            backbone_align = _run_usalign(backbone_pdb, folded_pdb)
            curve_align = _run_usalign(str(rec_curve_npy.with_suffix(".pdb")), str(source_curve_path.with_suffix(".pdb")))
            plddt = calculate_plddt(folded_pdb)["mean_plddt"]
            rec_curve = np.load(rec_curve_npy, allow_pickle=True).item()["curve_coords"]
            _, _, curve_sim = curve_similarity(torch.tensor(rec_curve), raw_curve_t)
            ss_acc = _compute_ss_accuracy(sketch_pdb, backbone_pdb) if sketch_pdb else 0.0
            folded_results.append(
                {
                    "folded_pdb": folded_pdb,
                    "backbone_pdb": backbone_pdb,
                    "rmsd_backbone": backbone_align["rmsd"],
                    "tm_backbone": backbone_align["tm"],
                    "rmsd_curve": curve_align["rmsd"],
                    "tm_curve": curve_align["tm"],
                    "plddt": float(plddt),
                    "curve_similarity": float(curve_sim),
                    "ss_acc": float(ss_acc),
                }
            )
        outputs.append(
            {
                "name": name,
                "backbone_pdb": backbone_pdb,
                "sketch_pdb": sketch_pdb,
                "folded_results": folded_results,
            }
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_bbs", type=int, default=1)
    parser.add_argument("--stage", choices=["backbone", "sequence_fold", "evaluation"], default="backbone")
    parser.add_argument("--num_seqs", type=int, default=4)
    parser.add_argument("--lmpnn_temperature", type=float, default=0.1)
    args = parser.parse_args()

    selection_manifest = Path(args.selection_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "downstream_summary.json"

    try:
        _ensure_import_paths()
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        os.chdir(output_root)

        if args.stage == "backbone":
            selection = _load_selection(selection_manifest)
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
                "sequence_outputs": [],
                "stages": {
                    "curve_selection": {"status": "done", "count": len(selection.get("curves", []))},
                    "sketch": {"status": "done", "count": len(sketch_paths)},
                    "backbone": {"status": "done", "count": len(backbone_outputs)},
                    "sequence": {"status": "planned", "count": 0},
                    "folded": {"status": "planned", "count": 0},
                    "evaluation": {"status": "planned", "count": 0},
                },
            }
            _write_summary(summary_path, summary)
        elif args.stage == "sequence_fold":
            summary = _load_summary(summary_path)
            if not summary:
                raise RuntimeError("downstream summary not found; run sketch/backbone first")
            sequence_outputs = _run_sequence_and_fold(
                summary=summary,
                output_root=output_root,
                device=device,
                num_seqs=args.num_seqs,
                temperature=args.lmpnn_temperature,
            )
            total_sequences = sum(len(item.get("sequences", [])) for item in sequence_outputs)
            total_folded = sum(len(item.get("folded_pdbs", [])) for item in sequence_outputs)
            if total_sequences <= 0 and total_folded <= 0:
                raise RuntimeError("sequence/folded stage produced no FASTA or folded PDB outputs")
            summary["sequence_outputs"] = sequence_outputs
            evaluation_outputs = _run_evaluation(summary, output_root)
            total_eval = sum(len(item.get("folded_results", [])) for item in evaluation_outputs)
            summary["status"] = "done"
            stages = summary.setdefault("stages", {})
            stages["sequence"] = {"status": "done", "count": total_sequences}
            stages["folded"] = {"status": "done", "count": total_folded}
            stages["evaluation"] = {"status": "done", "count": total_eval}
            summary["evaluation_outputs"] = evaluation_outputs
            _write_summary(summary_path, summary)
        else:
            summary = _load_summary(summary_path)
            if not summary:
                raise RuntimeError("downstream summary not found; run previous stages first")
            evaluation_outputs = _run_evaluation(summary, output_root)
            total_eval = sum(len(item.get("folded_results", [])) for item in evaluation_outputs)
            if total_eval <= 0:
                raise RuntimeError("evaluation produced no metric rows")
            summary["status"] = "done"
            summary["evaluation_outputs"] = evaluation_outputs
            stages = summary.setdefault("stages", {})
            stages["evaluation"] = {"status": "done", "count": total_eval}
            _write_summary(summary_path, summary)
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
