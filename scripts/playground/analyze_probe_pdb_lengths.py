#!/usr/bin/env python3
# coding: utf-8

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
'''
Example:

python scripts/analyze_probe_pdb_lengths.py \
  --pdb \
  /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbase-chainA \
  /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbeta-chainA \
  /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDfilter-chainA
  '''

# ----------------------------------------------------------------------
# Paths and config
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

# Path to prp-data env (change if different)
PRP_ENV_PREFIX = Path(
    "/public/home/zhangyangroup/chengshiz/run/20250717_prp-data/prp-data/.pixi/envs/default"
)

# Output directory for figures and stats
OUTPUT_DIR = REPO_ROOT / "pdb_length_analysis"

# Number of CPU workers for prp-data
PRP_WORKERS = 16


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Make a safe file-name component."""
    bad_chars = [" ", "/", "\\", ":", ";", ","]
    out = name
    for c in bad_chars:
        out = out.replace(c, "_")
    return out


def collect_pdb_groups(pdb_roots: List[str]) -> Dict[str, List[Path]]:
    """
    Collect PDB files grouped by top-level argument.

    Each argument:
      - if directory: group_name = directory name, all .pdb inside
      - if file: group_name = file stem
    Returns: dict[group_name] = list of PDB Paths
    """
    groups: Dict[str, List[Path]] = {}

    for arg in pdb_roots:
        root = Path(arg).resolve()
        if root.is_dir():
            group_name = root.name
            for fn in sorted(root.iterdir()):
                if fn.is_file() and fn.suffix.lower() == ".pdb":
                    groups.setdefault(group_name, []).append(fn)
        elif root.is_file():
            if root.suffix.lower() == ".pdb":
                group_name = root.stem
                groups.setdefault(group_name, []).append(root)
            else:
                print("[Warn] skip non-pdb file:", str(root))
        else:
            print("[Warn] path not found, skip:", str(root))

    if not groups:
        raise RuntimeError("No valid .pdb files collected from given paths.")

    print("[Groups]")
    for g, files in groups.items():
        print("  - {}: {} pdb files".format(g, len(files)))
    return groups


def run_prp_process_multi_pdb(pdb_files: List[Path], tmp_root: Path, workers: int) -> Dict[Path, Path]:
    """
    Run prp-data process once on a directory containing all PDB files.

    Returns:
        mapping: PDB Path -> curve_npy Path
    """
    env_prefix = PRP_ENV_PREFIX.resolve()

    pdb_input_dir = tmp_root / "pdb_input"
    curves_out_dir = tmp_root / "curves_out"
    pdb_input_dir.mkdir(parents=True, exist_ok=True)
    curves_out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    used_names = set()

    for idx, pdb_path in enumerate(pdb_files):
        pdb_path = pdb_path.resolve()
        base = pdb_path.name
        dest_name = base
        if dest_name in used_names:
            dest_name = "{:04d}__{}".format(idx, base)
        used_names.add(dest_name)

        dst_pdb = pdb_input_dir / dest_name
        if pdb_path != dst_pdb:
            shutil.copy2(str(pdb_path), str(dst_pdb))

        dest_stem = dst_pdb.stem
        mapping[pdb_path] = {"stem": dest_stem}

    cmd = [
        "mamba",
        "run",
        "-p",
        str(env_prefix),
        "prp-data",
        "process",
        "--input",
        str(pdb_input_dir),
        "--output",
        str(curves_out_dir),
        "--workers",
        str(int(workers)),
        "--device",
        "cpu",
        "--metadata",
        "probe_length_metadata.json",
    ]
    print("[PRP] Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("[PRP] stdout:\n", result.stdout)
        print("[PRP] stderr:\n", result.stderr)
        raise RuntimeError("prp-data process failed with code {}".format(result.returncode))

    print("[PRP] process finished. stdout:\n", result.stdout)

    npy_files = [p for p in curves_out_dir.iterdir() if p.suffix == ".npy"]
    if not npy_files:
        raise RuntimeError("No .npy produced by prp-data under {}".format(curves_out_dir))
    npy_files_sorted = sorted(npy_files, key=lambda x: x.name)

    pdb_to_npy: Dict[Path, Path] = {}
    for pdb_path, meta in mapping.items():
        stem = meta["stem"]
        candidate = curves_out_dir / (stem + ".npy")
        if candidate.is_file():
            pdb_to_npy[pdb_path] = candidate
            continue

        found = None
        for fn in npy_files_sorted:
            if fn.name.startswith(stem):
                found = fn
                break

        if found is None and npy_files_sorted:
            found = npy_files_sorted[0]

        if found is None:
            raise RuntimeError(
                "Could not find any .npy for PDB {} (stem={})".format(str(pdb_path), stem)
            )

        pdb_to_npy[pdb_path] = found

    print("[PRP] Resolved {} PDBs to curve npy files".format(len(pdb_to_npy)))
    return pdb_to_npy


def infer_curve_length(npy_path: Path) -> int:
    """
    Infer curve length (number of residues) from a curve .npy file.

    Supports both dict-wrapped arrays and plain ndarray.
    """
    arr = np.load(str(npy_path), allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        try:
            arr = arr.item()
        except Exception:
            pass

    length = None

    if isinstance(arr, dict):
        if "curve_coords" in arr:
            v = arr["curve_coords"]
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                length = int(v.shape[0])
        else:
            for v in arr.values():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    length = int(v.shape[0])
                    break
    elif isinstance(arr, np.ndarray) and arr.ndim >= 1:
        length = int(arr.shape[0])

    if length is None:
        raise RuntimeError("Cannot infer length from {}".format(str(npy_path)))

    return length


def plot_histograms(group_lengths: Dict[str, List[int]], out_dir: Path) -> None:
    """Plot combined and per-group histograms of curve length."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- combined figure ---
    plt.figure(figsize=(8, 6), dpi=140)
    all_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for idx, (g, lens) in enumerate(group_lengths.items()):
        if not lens:
            continue
        arr = np.asarray(lens, dtype=np.int32)
        plt.hist(
            arr,
            bins=20,
            alpha=0.5,
            label=g,
            color=all_colors[idx % len(all_colors)],
        )
    plt.xlabel("sequence length")
    plt.ylabel("count")
    plt.title("Length histograms for all probe groups")
    plt.legend()
    plt.tight_layout()
    combined_path = out_dir / "length_hist_all_groups.png"
    plt.savefig(str(combined_path))
    plt.close()
    print("[Fig] saved combined histogram:", combined_path)

    # --- per-group figures ---
    for g, lens in group_lengths.items():
        if not lens:
            continue
        arr = np.asarray(lens, dtype=np.int32)
        plt.figure(figsize=(8, 6), dpi=140)
        plt.hist(arr, bins=20, alpha=0.8)
        plt.xlabel("sequence length")
        plt.ylabel("count")
        plt.title("Length histogram for group: {}".format(g))
        plt.tight_layout()
        safe_name = sanitize_name(g)
        out_path = out_dir / f"length_hist_{safe_name}.png"
        plt.savefig(str(out_path))
        plt.close()
        print("[Fig] saved group histogram for {}: {}".format(g, out_path))


def write_stats(group_lengths: Dict[str, List[int]], out_dir: Path) -> None:
    """Write basic statistics and per-sample lengths to a text file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "length_stats.txt"

    with out_path.open("w") as f:
        for g, lens in group_lengths.items():
            f.write("Group: {}\n".format(g))
            if not lens:
                f.write("  (no samples)\n\n")
                continue
            arr = np.asarray(lens, dtype=np.float32)
            f.write("  count = {}\n".format(len(arr)))
            f.write("  min   = {:.1f}\n".format(arr.min()))
            f.write("  max   = {:.1f}\n".format(arr.max()))
            f.write("  mean  = {:.2f}\n".format(arr.mean()))
            f.write("  std   = {:.2f}\n".format(arr.std()))
            f.write("  lengths:\n")
            f.write("    " + ", ".join(str(int(x)) for x in arr) + "\n\n")

    print("[Stats] saved length stats to", out_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Analyze length distribution of probe PDBs after prp-data")
    p.add_argument(
        "--pdb",
        type=str,
        nargs="+",
        required=True,
        help="One or more PDB paths; each can be a file or a directory containing .pdb files",
    )
    return p.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[Out] results will be saved under:", str(OUTPUT_DIR))

    # 1) Collect PDBs and group names
    groups = collect_pdb_groups(args.pdb)

    # Flatten list of all PDBs
    all_pdb_files: List[Path] = []
    for files in groups.values():
        all_pdb_files.extend(files)

    if not all_pdb_files:
        raise RuntimeError("No PDB files found in any group.")

    # 2) Run prp-data once on all PDBs
    with tempfile.TemporaryDirectory(prefix="probe_len_") as tmp_root_str:
        tmp_root = Path(tmp_root_str).resolve()
        pdb_to_npy = run_prp_process_multi_pdb(
            pdb_files=all_pdb_files,
            tmp_root=tmp_root,
            workers=PRP_WORKERS,
        )

        # 3) Infer lengths for each group
        group_lengths: Dict[str, List[int]] = {g: [] for g in groups.keys()}

        for g, files in groups.items():
            for pdb_path in files:
                npy_path = pdb_to_npy.get(pdb_path)
                if npy_path is None:
                    print("[Warn] no npy found for PDB:", pdb_path)
                    continue
                try:
                    L = infer_curve_length(npy_path)
                except Exception as e:
                    print("[Warn] failed to infer length for {}: {}".format(npy_path, e))
                    continue
                group_lengths[g].append(L)

    # 4) Plot histograms
    plot_histograms(group_lengths, OUTPUT_DIR)

    # 5) Write stats
    write_stats(group_lengths, OUTPUT_DIR)

    print("[Done] Length analysis finished.")


if __name__ == "__main__":
    main()
