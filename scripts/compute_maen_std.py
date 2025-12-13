#!/usr/bin/env python3
import os
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, get_start_method

# ===== Fixed paths =====
NPY_DIR = Path("/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy/")
TRAIN_LIST = NPY_DIR / "train_list.txt"
PRECISION = 8          # decimals for printing
NUM_PROCESSES = 64     # <-- fixed to 64 processes

def _load_and_reduce_one(file_path: str):
    """
    Worker: load one .npy file, center per-curve, return partial sums.
    Returns (sum_xyz[3], sumsq_xyz[3], count:int)
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
        coords = np.asarray(data["curve_coords"], dtype=np.float64)  # [L,3]
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Bad coords shape at {file_path}: {coords.shape}")
        # per-curve centering
        coords = coords - coords.mean(axis=0, keepdims=True)

        s = coords.sum(axis=0)           # [3]
        ss = (coords ** 2).sum(axis=0)   # [3]
        c = coords.shape[0]
        return s, ss, c, None
    except Exception as e:
        return None, None, 0, f"{file_path}: {e}"

def _print_progress(done, total, start_t, last_printed, min_interval=0.2):
    now = time.time()
    if (now - last_printed[0]) < min_interval and done < total:
        return
    elapsed = now - start_t
    rate = (done / elapsed) if elapsed > 0 else 0.0
    pct = 100.0 * done / total if total > 0 else 0.0
    bar_len = 30
    filled = int(bar_len * done / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:6.2f}%  {done}/{total}  {rate:7.2f} files/s", end="", flush=True)
    last_printed[0] = now
    if done >= total:
        print()

def compute_stats_parallel(npy_dir: Path, train_list_file: Path, num_workers: int):
    with open(train_list_file, "r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]
    file_paths = [str(npy_dir / p) for p in rel_paths]
    n_files = len(file_paths)
    if n_files == 0:
        raise RuntimeError(f"No files listed in {train_list_file}")

    total_count = 0
    sum_xyz = np.zeros(3, dtype=np.float64)
    sumsq_xyz = np.zeros(3, dtype=np.float64)

    start_t = time.time()
    last_printed = [0.0]
    errors = []

    chunksize = max(1, n_files // (num_workers * 8))

    with Pool(processes=num_workers) as pool:
        done = 0
        for s, ss, c, err in pool.imap_unordered(_load_and_reduce_one, file_paths, chunksize=chunksize):
            if err is not None:
                errors.append(err)
            else:
                if c > 0:
                    sum_xyz += s
                    sumsq_xyz += ss
                    total_count += c
            done += 1
            _print_progress(done, n_files, start_t, last_printed)

    if errors:
        print("\n[Warnings] Some files failed to process:")
        for e in errors[:10]:
            print("  -", e)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if total_count == 0:
        raise RuntimeError("No points accumulated (all files failed or empty).")

    mean_xyz = sum_xyz / total_count
    var_xyz = (sumsq_xyz / total_count) - (mean_xyz ** 2)
    var_xyz = np.maximum(var_xyz, 0.0)
    std_xyz = np.sqrt(var_xyz)

    return mean_xyz.astype(np.float64), std_xyz.astype(np.float64)

def main():
    if not TRAIN_LIST.is_file():
        raise FileNotFoundError(f"Missing train_list.txt at {TRAIN_LIST}")

    print(f"[Info] Using npy_dir:   {NPY_DIR}")
    print(f"[Info] Using train_list: {TRAIN_LIST}")
    print(f"[Info] Processes:        {NUM_PROCESSES}")

    mean_xyz, std_xyz = compute_stats_parallel(NPY_DIR, TRAIN_LIST, NUM_PROCESSES)

    fmt = f"{{:.{PRECISION}f}}"
    mean_str = "[" + ", ".join(fmt.format(x) for x in mean_xyz.tolist()) + "]"
    std_str  = "[" + ", ".join(fmt.format(x) for x in std_xyz.tolist()) + "]"

    print("\n# Paste into configs YAML under data_params")
    print(f"mean_xyz: {mean_str}")
    print(f"std_xyz:  {std_str}")

if __name__ == "__main__":
    try:
        get_start_method()
    except RuntimeError:
        pass
    main()
