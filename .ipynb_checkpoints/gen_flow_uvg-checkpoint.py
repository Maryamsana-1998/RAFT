import os
import glob
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import flowiz as fz
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# === Config ===
videos = ["Beauty", 
    # "HoneyBee", 
    "ReadySteadyGo", 
    "YachtRide", 
    # "Bosphorus", 
    "Jockey", 
    #"ShakeNDry"
    ]

res = "1080p"
gop_list = [4, 8, 16]
gpu_count = 1 # we have 2 GPUs
max_workers = 8  # 8 CPUs per GPU * 2 GPUs
base_dir = "/data/maryam.sana/Ultra_Perceptual_Video_Compression/data/UVG"

# === Optical Flow Script Parameters ===
def compute_flow(r1_path, r2_path, flo_output_path, png_output_path, gpu_id):
    try:
        command = [
            "python3", "run.py",
            "--model", "sintel-final",
            "--one", r1_path,
            "--two", r2_path,
            "--out", flo_output_path
        ]
        # Set environment variable for CUDA
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run(command, env=env, check=True)

        # Convert flow file to an image
        flow_img = fz.convert_from_file(flo_output_path)
        plt.imsave(png_output_path, flow_img)
        print(f"Flow computed: {os.path.basename(flo_output_path)} on GPU {gpu_id}", flush=True)
    except Exception as e:
        print(f"Error on {flo_output_path}: {e}", flush=True)

# === Worker Function ===
def process_flow_task(task):
    ref_frame, tgt_frame, flo_output_path, png_output_path, gpu_id = task
    # Skip processing if decoded flow image already exists
    if Path(png_output_path).exists():
        return f"Skipped {png_output_path} (already exists)"
    compute_flow(ref_frame, tgt_frame, flo_output_path, png_output_path, gpu_id)
    size_kb = os.path.getsize(flo_output_path) / 1024 if os.path.exists(flo_output_path) else 0
    return f"Processed {os.path.basename(flo_output_path)}: {size_kb:.2f} KB on GPU {gpu_id}"

# === Main Processing ===
def main():
    tasks = []
    # For each video and each GOP value, create separate output folders
    for video in videos:
        for gop in gop_list:
            print(f"\n🚀 Processing {video} [{res}] with GOP={gop}")
            input_folder = os.path.join(base_dir, video, res)
            flow_folder = os.path.join(base_dir, video, res, f"optical_flow_gop_{gop}")
            os.makedirs(flow_folder, exist_ok=True)
            
            frame_paths = sorted(glob.glob(os.path.join(input_folder, "*.png")))
            total_frames = len(frame_paths)
            # Create tasks per GOP chunk
            for i in range(0, total_frames, gop):
                if i + gop >= total_frames:
                    break  # no full GOP left
                ref_idx = i
                ref_frame = frame_paths[ref_idx]
                for offset in range(1, gop):
                    tgt_idx = ref_idx + offset
                    tgt_frame = frame_paths[tgt_idx]
                    # Define output file names
                    base_name = f"flow_{ref_idx:04d}_{tgt_idx:04d}"
                    flo_output_path = os.path.join(flow_folder, base_name + ".flo")
                    png_output_path = os.path.join(flow_folder, base_name + ".png")
                    # Assign GPU in a round-robin fashion
                    task_index = len(tasks)
                    gpu_id = task_index % gpu_count
                    tasks.append((ref_frame, tgt_frame, flo_output_path, png_output_path, gpu_id))

    print(f"\nTotal tasks: {len(tasks)}")
    
    # Use ProcessPoolExecutor with tqdm for progress bar
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_flow_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Optical Flow"):
            result = future.result()  # Optionally process or log result
            # You can print the result if desired:
            # print(result)

    print("\n🎉 All optical flow computations completed.")

if __name__ == "__main__":
    main()
