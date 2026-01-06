import numpy as np

npy_path = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/prior_recons_300/sample_prior_0000_recon.npy"
txt_path = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/recon_epoch_49_0_full_dump.txt"

data = np.load(npy_path, allow_pickle=True)

with open(txt_path, 'w', encoding='utf-8') as f:
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 6:
        f.write(f"Invalid shape: {data.shape}, expected (L, 6)\n")
        exit(1)

    mask = np.any(data[:, :3] != 0, axis=1)
    valid_curve = data[mask]

    f.write(f"Curve shape: {data.shape}\n")
    f.write(f"Valid points: {valid_curve.shape[0]}\n")
    f.write("Index |     X       Y       Z     |    SS One-hot\n")
    f.write("-" * 60 + "\n")

    for i in range(valid_curve.shape[0]):
        xyz = valid_curve[i, :3]
        ss = valid_curve[i, 3:]
        xyz_str = " ".join(f"{x:8.3f}" for x in xyz)
        ss_str = " ".join(f"{s:.3f}" for s in ss)
        f.write(f"{i:5d} | {xyz_str} | {ss_str}\n")

print(f"? Saved full content of single curve to {txt_path}")
