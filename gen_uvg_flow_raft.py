#!/usr/bin/env python
# compute_flows_gop.py
# -----------------------------------------------------------
import os, glob, argparse, struct
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch
import numpy as np
import cv2

# ------------- import RAFT + helpers -----------------------
from raft import RAFT
from utils.utils import InputPadder
# -----------------------------------------------------------

# -----------------------------------------------------------
# Utilities you already provided (kept unchanged)
# -----------------------------------------------------------
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
    img = cv2.resize(img, (512,512))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)

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

    for flow, unpad, out_path in zip(flows, unpads, out_paths):
        flow = unpad(flow[None])[0].permute(1, 2, 0).cpu().numpy()
        write_flo_file(flow, out_path)

# ----------------- CONFIG ----------------------------------
VIDEOS   = ["Beauty", "HoneyBee", "ReadySteadyGo", "YachtRide",
            "Bosphorus", "Jockey", "ShakeNDry"]
GOPS     = [2]
BASE_DIR = "/data/maryam.sana/Ultra_Perceptual_Video_Compression/data/UVG"
CKPT     = "models/raft-sintel.pth"             # <-- put RAFT ckpt here
GPU_COUNT   = 1                                   # e.g. 2 GPUs
MAX_WORKERS = 8                                   # CPU workers per job
BATCH_SIZE  = 8                                  # flow pairs per RAFT forward
# RES_FOLDER  = "512x512"                           # images are already 512x512
# -----------------------------------------------------------

def build_tasks() -> List[Tuple[str,str,str,int]]:
    """Return list of (ref, tgt, flo_out, gpu_id) tuples."""
    tasks = []
    for video in VIDEOS:
        for gop in GOPS:
            img_dir = Path(BASE_DIR) / video / "images"
            flow_dir = Path(BASE_DIR) / video / "optical_flow" / f"optical_flow_gop_{gop}_raft"
            flow_dir.mkdir(parents=True, exist_ok=True)

            frames = sorted(img_dir.glob("*.png"))
            total = len(frames)

            for i in range(0, total, gop):
                if i + gop >= total:
                    break                     # skip incomplete GOP
                ref = frames[i]               # anchor frame
                for off in range(1, gop):
                    tgt = frames[i + off]
                    name = f"flow_{i:04d}_{i+off:04d}"
                    flo_out = flow_dir / f"{name}.flo"
                    # round-robin GPU
                    gpu_id = len(tasks) % GPU_COUNT
                    tasks.append((str(ref), str(tgt), str(flo_out), gpu_id))
    return tasks

# -------------- Worker wrapper -----------------------------
def flow_worker(batch: List[Tuple[str,str,str,int]]) -> str:
    """Compute flows for a list of pairs on the assigned GPU."""
    gpu_id = batch[0][3]                         # all pairs share same GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # load RAFT once per worker
    raft_args = argparse.Namespace(small=False, mixed_precision=False,
                                   alternate_corr=False, model=CKPT)
    model = RAFT(raft_args)
    state = torch.load(CKPT, map_location=device)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # prepare batch lists
    pairs, outs = [], []
    for ref, tgt, flo_out, _ in batch:
        if Path(flo_out).exists():
            continue
        pairs.append((ref, tgt))
        outs.append(flo_out)

    if pairs:
        process_batch(model, pairs, outs, device)
    return f"GPU{gpu_id}: {len(pairs)} flows written."

# -------------- Main ---------------------------------------
def main():
    tasks = build_tasks()
    if not tasks:
        print("No work to do. Exiting.")
        return

    # group tasks into mini-batches
    batched_tasks = []
    for i in range(0, len(tasks), BATCH_SIZE):
        batched_tasks.append(tasks[i:i+BATCH_SIZE])

    print(f"Total flow pairs: {len(tasks)} | Batches: {len(batched_tasks)}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = [exe.submit(flow_worker, batch) for batch in batched_tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Optical-Flow"):
            _ = f.result()  # could log if needed

    print("🎉 All optical-flow computations completed.")

if __name__ == "__main__":
    main()
