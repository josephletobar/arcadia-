import argparse
import importlib.util
import json
import os
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.4"


def _number(value, default=0.0):
    if value is None:
        return default

    if isinstance(value, bytes):
        try:
            import numpy as np

            for dtype in (np.float64, np.float32, np.int64, np.int32):
                decoded = np.frombuffer(value, dtype=dtype)
                if decoded.size:
                    return float(decoded[0])
        except Exception:
            return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _require_columns(cursor, table_name, required):
    columns = _table_columns(cursor, table_name)
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(
            f"{table_name!r} is missing required column(s): {', '.join(missing)}"
        )
    return columns


def _connect_read_only(db_path):
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    return sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)


def _area_label(x, y, bounds):
    min_x, max_x, min_y, max_y = bounds
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    x_norm = (x - min_x) / x_span
    y_norm = (y - min_y) / y_span

    if x_norm < 1 / 3:
        horizontal = "left"
    elif x_norm < 2 / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    if y_norm < 1 / 3:
        vertical = "upper"
    elif y_norm < 2 / 3:
        vertical = "middle"
    else:
        vertical = "lower"

    if horizontal == "center" and vertical == "middle":
        return "central area"
    if horizontal == "center":
        return f"{vertical} central area"
    if vertical == "middle":
        return f"{horizontal} middle area"
    return f"{vertical} {horizontal} area"


def load_scene_db(db_path):
    conn = _connect_read_only(db_path)
    try:
        cursor = conn.cursor()
        if not _table_exists(cursor, "nodes"):
            raise ValueError("DB does not contain a nodes table")

        node_columns = _require_columns(
            cursor,
            "nodes",
            ["id", "label", "score", "count", "geo_pos_x", "geo_pos_y"],
        )
        reasoning_select = "reasoning" if "reasoning" in node_columns else "''"
        cursor.execute(
            f"""
            SELECT id, label, score, count, geo_pos_x, geo_pos_y, {reasoning_select}
            FROM nodes
            ORDER BY rowid
            """
        )
        node_rows = cursor.fetchall()
        positions = [
            (_number(row[4]), _number(row[5]))
            for row in node_rows
        ]
        if positions:
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]
            bounds = (min(xs), max(xs), min(ys), max(ys))
        else:
            bounds = (0.0, 1.0, 0.0, 1.0)

        nodes = []
        for node_id, label, score, count, x, y, reasoning in node_rows:
            x = _number(x)
            y = _number(y)
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "score": _number(score),
                    "count": int(_number(count, default=1)),
                    "position": {
                        "x": x,
                        "y": y,
                    },
                    "area": _area_label(x, y, bounds),
                    "reasoning": reasoning or "",
                }
            )

        edges = []
        if _table_exists(cursor, "edges"):
            _require_columns(cursor, "edges", ["source_id", "target_id", "weight"])
            cursor.execute(
                """
                SELECT source_id, target_id, weight
                FROM edges
                ORDER BY source_id, target_id
                """
            )
            edges = [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "distance": _number(weight),
                }
                for source_id, target_id, weight in cursor.fetchall()
            ]

        return {
            "nodes": nodes,
            "spatial_edges": edges,
        }
    finally:
        conn.close()


def build_prompt(scene_graph, question):
    graph_json = json.dumps(scene_graph, indent=2)
    return f"""
You are answering questions about a saved drone scene graph.

Scene graph:
{graph_json}

Question:
{question}

Answer style:
- Be direct.
- 3 sentance max.
- Use coordinates internally for spatial reasoning, but do not print raw coordinate
  pairs or coordinate lists in the answer.
- Also do not output high-level spatial informaton like "middle of the scene", "northwest corner", "right side" etc, 
    there is no notion of absolute direction or centrality in the graph.
- Prefer clean conceptual scene areas and actions over numeric location dumps.
- You may refer to broad areas such as
  road-connected area, building compound, field edge, forested area, or open yard
  only when supported by the graph.
- Give a short recommendation and brief reasoning based on node scores, node
  reasoning, spatial layout, and spatial edges.
- If the graph is missing enough evidence, say that plainly.
- Do not invent objects, locations, roads, buildings, or relationships not present
  in the graph.
""".strip()


def ask_scene(client, scene_graph, question):
    response = client.responses.create(
        model=MODEL,
        input=build_prompt(scene_graph, question),
    )
    return response.output_text.strip()


def _viewer_python():
    """Find a Python environment containing the reconstruction viewer."""
    if importlib.util.find_spec("colmap_reconstruction") and importlib.util.find_spec("pyvista"):
        return Path(sys.executable)

    try:
        conda_base = Path(
            subprocess.check_output(
                ["conda", "info", "--base"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        conda_base = None

    if conda_base is not None:
        candidate = conda_base / "envs" / "colmap-reconstruction" / "python.exe"
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Could not find the colmap-reconstruction Python environment needed "
        "to display the optional PLY mesh."
    )


def desktop_work_area():
    """Return the usable desktop rectangle as x, y, width, height."""
    if sys.platform != "win32":
        return 0, 0, 1800, 900
    import ctypes

    class Rect(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = Rect()
    if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, 1800, 900


def _visible_window_handles():
    if sys.platform != "win32":
        return set()
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    handles = set()
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
            handles.add(int(hwnd))
        return True

    user32.EnumWindows(callback_type(visit), 0)
    return handles


def _move_process_window(
    process,
    geometry,
    timeout=45.0,
    title=None,
    excluded_handles=(),
):
    """Wait for a process's visible window and move it to a desktop rectangle."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    x, y, width, height = geometry
    user32 = ctypes.windll.user32
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _lparam):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title_length = user32.GetWindowTextLengthW(hwnd)
        window_title = ""
        if title_length:
            buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, buffer, len(buffer))
            window_title = buffer.value
        matches_process = pid.value == process.pid
        matches_new_title = (
            title is not None
            and window_title == title
            and int(hwnd) not in excluded_handles
        )
        if user32.IsWindowVisible(hwnd) and (matches_process or matches_new_title):
            found.append(hwnd)
            return False
        return True

    callback = callback_type(visit)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process.poll() is None:
        found.clear()
        user32.EnumWindows(callback, 0)
        if found:
            user32.MoveWindow(found[0], x, y, width, height, True)
            return True
        time.sleep(0.1)
    return False


def _position_own_console(geometry):
    if sys.platform != "win32":
        return
    import ctypes

    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.MoveWindow(hwnd, *geometry, True)


def launch_mesh_viewer(mesh_path, db_path=None):
    mesh_path = Path(mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"PLY mesh not found: {mesh_path}")
    if mesh_path.suffix.lower() != ".ply":
        raise ValueError(f"Mesh must be a .ply file: {mesh_path}")
    command = [
        str(_viewer_python()),
        "-m",
        "colmap_reconstruction.paint_heatmap",
        str(mesh_path),
    ]
    if db_path is not None:
        command.append(str(Path(db_path).resolve()))
    return subprocess.Popen(command)


def _close_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


def run_chat(db_path):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    scene_graph = load_scene_db(db_path)

    print("\nScene DB chat started. Type 'quit' or 'exit' to leave.")
    print(
        f"Loaded {len(scene_graph['nodes'])} node(s), "
        f"{len(scene_graph['spatial_edges'])} spatial edge(s)."
    )

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        # Painting commits directly to graph.db, so reload before every query.
        scene_graph = load_scene_db(db_path)
        print("\nAssistant:")
        print(ask_scene(client, scene_graph, question))


def run_graph_window(db_path, geometry):
    """Show Priority Map's exact spatial renderer and follow SQLite commits."""
    import matplotlib.pyplot as plt
    from priority_map.modules.GraphBuilder import GraphBuilder

    db_path = Path(db_path).resolve()
    if db_path.name.lower() != "graph.db":
        raise ValueError("The spatial viewer expects a Priority Map file named graph.db")

    builder = GraphBuilder(output_dir=db_path.parent)
    figure, axis = plt.subplots(figsize=(8, 5))
    manager = figure.canvas.manager
    if hasattr(manager, "set_window_title"):
        manager.set_window_title("Priority Map - Spatial Knowledge Graph")
    window = getattr(manager, "window", None)
    x, y, width, height = geometry
    if hasattr(window, "wm_geometry"):
        window.wm_geometry(f"{width}x{height}+{x}+{y}")
    elif hasattr(window, "setGeometry"):
        window.setGeometry(x, y, width, height)

    def redraw():
        axis.clear()
        builder.draw_2d_graph_axes(axis, view="spatial")
        figure.canvas.draw_idle()
        figure.canvas.flush_events()

    try:
        redraw()
        data_version = builder.conn.execute("PRAGMA data_version").fetchone()[0]
        manager.show()
        while plt.fignum_exists(figure.number):
            current_version = builder.conn.execute("PRAGMA data_version").fetchone()[0]
            if current_version != data_version:
                data_version = current_version
                redraw()
            plt.pause(0.25)
    finally:
        builder.close()


def _child_command(mode, db_path, geometry):
    x, y, width, height = geometry
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        mode,
        "--graph-db",
        str(Path(db_path).resolve()),
        "--_x",
        str(x),
        "--_y",
        str(y),
        "--_width",
        str(width),
        "--_height",
        str(height),
    ]


def launch_graph(db_path, geometry):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        _child_command("graph", db_path, geometry),
        creationflags=flags,
    )


def launch_chat(db_path, geometry):
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return subprocess.Popen(
        _child_command("chat", db_path, geometry),
        creationflags=flags,
    )


def run_workspace(mesh_path, graph_db=None, chat=False):
    mesh_path = Path(mesh_path).resolve()
    graph_db = Path(graph_db).resolve() if graph_db is not None else None
    if not mesh_path.is_file():
        raise FileNotFoundError(f"PLY mesh not found: {mesh_path}")
    if mesh_path.suffix.lower() != ".ply":
        raise ValueError(f"Mesh must be a .ply file: {mesh_path}")
    if graph_db is not None and not graph_db.is_file():
        raise FileNotFoundError(f"Graph DB not found: {graph_db}")
    if chat and graph_db is None:
        raise ValueError("--chat requires --graph-db")

    left, top, width, height = desktop_work_area()
    children = []
    existing_windows = _visible_window_handles()
    painter = launch_mesh_viewer(mesh_path, graph_db)
    try:
        if graph_db is None:
            painter_geometry = (left, top, width, height)
        else:
            painter_width = width // 2
            painter_geometry = (left, top, painter_width, height)
            right_x = left + painter_width
            right_width = width - painter_width
            if chat:
                graph_height = height // 2
                graph_geometry = (right_x, top, right_width, graph_height)
                chat_geometry = (
                    right_x,
                    top + graph_height,
                    right_width,
                    height - graph_height,
                )
                children.append(launch_chat(graph_db, chat_geometry))
            else:
                graph_geometry = (right_x, top, right_width, height)
            children.append(launch_graph(graph_db, graph_geometry))

        if not _move_process_window(
            painter,
            painter_geometry,
            title=f"Heatmap Painter - {mesh_path.name}",
            excluded_handles=existing_windows,
        ):
            print("Warning: could not position the painter window.", file=sys.stderr)
        while painter.poll() is None:
            time.sleep(0.25)
        return painter.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        for process in reversed(children):
            _close_process(process)
        _close_process(painter)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open an Arcadia mesh workspace with optional graph and chat."
    )
    parser.add_argument("mesh", nargs="?", type=Path, help="PLY mesh to open in the painter.")
    parser.add_argument(
        "--graph-db",
        type=Path,
        help="Optional Priority Map graph.db for DB painting and spatial graph display.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Open Scene Chat below the spatial graph; requires --graph-db.",
    )
    parser.add_argument("--_child", choices=("graph", "chat"), help=argparse.SUPPRESS)
    parser.add_argument("--_x", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_y", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_width", type=int, default=900, help=argparse.SUPPRESS)
    parser.add_argument("--_height", type=int, default=900, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    geometry = (args._x, args._y, args._width, args._height)
    if args._child == "graph":
        if args.graph_db is None:
            raise SystemExit("Graph child requires --graph-db")
        run_graph_window(args.graph_db, geometry)
        return
    if args._child == "chat":
        if args.graph_db is None:
            raise SystemExit("Chat child requires --graph-db")
        _position_own_console(geometry)
        run_chat(args.graph_db)
        return
    if args.mesh is None:
        raise SystemExit("A mesh path is required")
    try:
        raise SystemExit(run_workspace(args.mesh, args.graph_db, chat=args.chat))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
