#!/usr/bin/env python
"""
Batched RAFT optical‑flow extractor (PyTorch 1.6‑safe)
=====================================================
* Loads RAFT **once per GPU** inside each worker.
* Runs **mini‑batch inference** (`--batch-size`, default 8).
* Uses the **spawn** multiprocessing start‑method to avoid CUDA context clashes in old PyTorch.
* No modern syntax that PyTorch 1.6 / Python 3.8 can’t parse.
"""

import os
import argparse
import struct
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple
from tqdm import tqdm

# RAFT + helpers --------------------------------------------------------------
from raft import RAFT                      # <‑ ensure RAFT repo in PYTHONPATH
from utils.utils import InputPadder         # <‑ from official RAFT utils

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def write_flo_file(flow: np.ndarray, filename: str) -> None:
    """Save optical flow to Middlebury .flo"""
    assert flow.ndim == 3 and flow.shape[2] == 2, "Flow must have shape (H, W, 2)"
    magic = 202021.25
    h, w = flow.shape[:2]
    with open(filename, "wb") as f:
        f.write(struct.pack("f", magic))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        f.write(flow.astype(np.float32).tobytes())


def load_image(path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)

# -----------------------------------------------------------------------------
# Core batch routine
# -----------------------------------------------------------------------------

def process_batch(model: torch.nn.Module,
                  pairs: List[Tuple[str, str]],
                  out_paths: List[str],
                  device: torch.device) -> None:
    imgs1, imgs2, unpads = [], [], []
    for ref_path, tgt_path in pairs:
        ref = load_image(ref_path, device)
        tgt = load_image(tgt_path, device)
        padder = InputPadder(ref.shape)
        ref, tgt = padder.pad(ref, tgt)
        imgs1.append(ref)
        imgs2.append(tgt)
        unpads.append(padder.unpad)

    imgs1 = torch.cat(imgs1, dim=0)
    imgs2 = torch.cat(imgs2, dim=0)

    with torch.no_grad():
        _, flows = model(imgs1, imgs2, iters=20, test_mode=True)

    # Write each flow
    for flow, unpad, out_path in zip(flows, unpads, out_paths):
        flow = unpad(flow[None])[0].permute(1, 2, 0).cpu().numpy()
        write_flo_file(flow, out_path)

# -----------------------------------------------------------------------------
# Worker process (one per GPU)
# -----------------------------------------------------------------------------

def worker(gpu_id: int,
           folders: List[str],
           ckpt: str,
           batch: int,
           overwrite: bool) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # isolate GPU for this proc
    device = torch.device("cuda:0")

    # Load model **after** setting device
    raft_args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False, model=ckpt)
    model = RAFT(raft_args)
    state_dict = torch.load(ckpt, map_location=device)
    # Remove 'module.' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False) 
    # model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    for folder in folders:
        ref_path = os.path.join(folder, "r2.png")
        if not os.path.exists(ref_path):
            print("[WARN] Missing r2.png in", folder)
            continue

        flow_dir = os.path.join(folder, "Flow_b")
        os.makedirs(flow_dir, exist_ok=True)

        images = [f for f in sorted(os.listdir(folder))
                  if f.lower().endswith((".png", ".jpg"))]

        pairs, outs = [], []
        for img in images:
            tgt_path = os.path.join(folder, img)
            out_path = os.path.join(flow_dir, img.rsplit(".", 1)[0] + ".flo")
            if not overwrite and os.path.exists(out_path):
                continue
            pairs.append((ref_path, tgt_path))
            outs.append(out_path)

        # Mini‑batch
        for i in range(0, len(pairs), batch):
            try:
                process_batch(model, pairs[i:i + batch], outs[i:i + batch], device)
                print('batch processed bwd')
            except Exception as exc:
                print(f"[GPU {gpu_id}] Error in {folder}: {exc}")

# -----------------------------------------------------------------------------
# Utility: split folders across GPUs
# -----------------------------------------------------------------------------

def shard(folders: List[str], n: int) -> List[List[str]]:
    buckets = [[] for _ in range(n)]
    for idx, fld in enumerate(folders):
        buckets[idx % n].append(fld)
    return buckets

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------

def discover_folders(root: str) -> List[str]:
    out: List[str] = []
    for dpath, _dirs, files in os.walk(root):
        if "r2.png" in files:
            out.append(dpath)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    folders = discover_folders(args.root)
    print(f"Found {len(folders)} folders containing r2.png")

    shards = shard(folders, len(args.gpus))
    procs = []
    for gpu, shard_folds in zip(args.gpus, shards):
        p = mp.Process(target=worker,
                       args=(gpu, shard_folds, args.ckpt, args.batch_size, args.overwrite))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    # PyTorch 1.6 on Linux defaults to "fork" which is unsafe for CUDA; force spawn.
    mp.set_start_method("spawn", force=True)
    main()
