from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from priority_map import run_priority_map
from colmap_reconstruction import ReconstructionResult, reconstruct, project_heatmaps

# image_frames = Path(r"D:\dronevid2")
# image_frames = Path(r"D:\UAV_VisLoc_dataset\05\drone")
# image_frames = Path(r"D:\Drone subset 2")
# image_frames = Path(r"D:\dronevid2")
image_frames = Path(r"C:\Users\jletobar3\Desktop\dronevids")
arcadia_out = Path(r"C:/Users/jletobar3/Projects/arcadia/out10")

priority_out = arcadia_out / "priority_map"
colmap_out = arcadia_out / "colmap"

priority_out.mkdir(parents=True, exist_ok=True)
colmap_out.mkdir(parents=True, exist_ok=True)

# 1) Make heatmaps and reconstruct the same images at the same time.
def run_priority():
    return run_priority_map(
        image_folder=image_frames,
        output_dir=priority_out,
        task="Find cars",
        sam_step=30,
        debug=False,
        record=True,
        sam_model_path=Path(r"C:/Users/jletobar3/Projects/drone_heatmap/models/sam3.pt"),
    )


def run_colmap():
    return reconstruct(
        input_path=image_frames,
        output_dir=colmap_out,
        skip_frames=0,
        max_image_size=640,
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
