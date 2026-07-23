from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from priority_map import run_priority_map
from colmap_reconstruction import ReconstructionResult, reconstruct, project_heatmaps

parser = ArgumentParser(description="Run Arcadia heatmap + COLMAP pipeline")
parser.add_argument("--image-frames", required=True, type=Path, help="Directory containing input images/video frames")
parser.add_argument("--arcadia-out", required=True, type=Path, help="Directory for all outputs")
parser.add_argument(
    "--sam-model-path",
    required=True,
    type=Path,
    help="Path to SAM model checkpoint (e.g. sam3.pt)",
)
parser.add_argument("--sam-step", type=int, default=30, help="Sampling step for priority map generation")
parser.add_argument("--task", default="Find cars", help="Detection/classification prompt")
parser.add_argument("--debug", action="store_true", default=True, help="Enable priority map debug output")
parser.add_argument(
    "--no-record",
    dest="record",
    action="store_false",
    help="Disable heatmap recording in priority mapping",
)
parser.add_argument("--skip-frames", type=int, default=0, help="COLMAP skip-frames value")
parser.add_argument("--max-image-size", type=int, default=640, help="COLMAP max image size")
args = parser.parse_args()

image_frames = args.image_frames
arcadia_out = args.arcadia_out
sam_model_path = args.sam_model_path
sam_step = args.sam_step
task = args.task
debug = args.debug
record = args.record
skip_frames = args.skip_frames
max_image_size = args.max_image_size

priority_out = arcadia_out / "priority_map"
colmap_out = arcadia_out / "colmap"

priority_out.mkdir(parents=True, exist_ok=True)
colmap_out.mkdir(parents=True, exist_ok=True)

# 1) Make heatmaps and reconstruct the same images at the same time.
def run_priority():
    return run_priority_map(
        image_folder=image_frames,
        output_dir=priority_out,
        task=task,
        sam_step=sam_step,
        debug=debug,
        record=record,
        sam_model_path=sam_model_path,
    )


def run_colmap():
    return reconstruct(
        input_path=image_frames,
        output_dir=colmap_out,
        skip_frames=skip_frames,
        max_image_size=max_image_size,
    )


if image_frames.exists():
    with ThreadPoolExecutor(max_workers=2) as executor:
        priority_future = executor.submit(run_priority)
        colmap_future = executor.submit(run_colmap)

        priority_result = priority_future.result()
        reconstruction = colmap_future.result()

    print(priority_result)
    print(reconstruction.mesh_path)
else:
    print(f"Set image_frames first: {image_frames}")

# 3) Project priority-map heatmaps onto an existing reconstruction.
heatmap_dir = priority_out / "heatmap_imgs"

if colmap_out.exists() and heatmap_dir.exists():
    reconstruction = ReconstructionResult.from_output_dir(colmap_out)
    heatmapped = project_heatmaps(reconstruction, heatmap_dir)
    print(heatmapped.output_mesh_path)
else:
    print("Need colmap_out and priority_out/heatmap_imgs first")
