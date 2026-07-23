# arcadia

Arcadia runs a two-stage drone-vision pipeline: it generates priority heatmaps and a COLMAP reconstruction in parallel, then projects heatmaps onto the reconstructed model.

## Quick start (CLI)

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run Arcadia with explicit paths:

```bash
python arcadia.py `
  --image-frames "C:\path\to\frames" `
  --arcadia-out "C:\path\to\arcadia_out" `
  --sam-model-path "C:\path\to\sam3.pt"
```

Optional flags:
- `--sam-step` (default: 30)
- `--task` (default: `Find cars`)
- `--skip-frames` (default: 0)
- `--max-image-size` (default: 640)
- `--no-record` to disable heatmap recording (recording is enabled by default)
- `--debug` to enable verbose priority-map debug output

## Example CLI run

```bash
python arcadia.py --image-frames "C:\Users\jletobar3\Projects\query_images" --arcadia-out "C:\Users\jletobar3\Projects\arcadia\out12" --sam-model-path "C:\Users\jletobar3\Projects\drone_heatmap\models\sam3.pt"
```

Outputs:
- `out11\priority_map`
- `out11\colmap`
- projected mesh path is printed to console when projection runs successfully.

## Interactive scene workspace

Open a PLY in the standalone painter:

```powershell
python arcadia_view.py "C:\path\to\mesh.ply"
```

Pass a Priority Map database to synchronize painting. This does not display the
graph by default:

```powershell
python arcadia_view.py "C:\path\to\object_pins_on_mesh.ply" --graph-db "C:\path\to\graph.db"
```

Add `--graph-viz` to display the live spatial graph beside the painter:

```powershell
python arcadia_view.py "C:\path\to\object_pins_on_mesh.ply" --graph-db "C:\path\to\graph.db" --graph-viz
```

Add `--chat` to open Scene Chat beside the painter. Chat reloads `graph.db`
before each question so answers reflect committed painting changes:

```powershell
python arcadia_view.py "C:\path\to\object_pins_on_mesh.ply" --graph-db "C:\path\to\graph.db" --chat
```

Use both flags to stack the graph and chat on the right:

```powershell
python arcadia_view.py "C:\path\to\object_pins_on_mesh.ply" --graph-db "C:\path\to\graph.db" --graph-viz --chat
```

#### Local Example

```powershell
cd "C:\Users\jletobar3\Projects\arcadia"

python arcadia_view.py "C:\Users\jletobar3\Projects\smallout\object_pins\object_pins_on_mesh.ply" --graph-db "C:\Users\jletobar3\Projects\drone_heatmap\examples\2026-07-13_12-45-06\graph.db" --chat --graph-viz
```
