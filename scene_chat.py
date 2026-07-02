import argparse
import json
import os
import sqlite3
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
- Be crisp and direct.
- Use coordinates internally for spatial reasoning, but do not print raw coordinate
  pairs or coordinate lists in the answer.
- Prefer clean conceptual scene areas and actions over numeric location dumps.
- You may refer to broad areas such as upper-left, central area, lower-right,
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
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        print("\nAssistant:")
        print(ask_scene(client, scene_graph, question))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat crisply with a saved drone scene graph DB."
    )
    parser.add_argument("db_path", help="Path to graph.db.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_chat(args.db_path)


if __name__ == "__main__":
    main()
