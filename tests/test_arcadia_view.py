from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, call, patch

import arcadia_view


class ArcadiaViewTests(unittest.TestCase):
    def test_chat_requires_graph_database(self):
        with TemporaryDirectory() as temp_dir:
            mesh = Path(temp_dir) / "mesh.ply"
            mesh.touch()
            with self.assertRaisesRegex(ValueError, "--chat requires --graph-db"):
                arcadia_view.run_workspace(mesh, chat=True)

    def test_workspace_stacks_graph_and_chat_to_the_right(self):
        with TemporaryDirectory() as temp_dir:
            mesh = Path(temp_dir) / "mesh.ply"
            db = Path(temp_dir) / "graph.db"
            mesh.touch()
            db.touch()
            painter = MagicMock()
            painter.pid = 10
            painter.poll.side_effect = [None, 0]
            painter.returncode = 0
            graph = MagicMock()
            graph.poll.return_value = None
            chat = MagicMock()
            chat.poll.return_value = None

            with (
                patch.object(arcadia_view, "desktop_work_area", return_value=(0, 0, 1200, 800)),
                patch.object(arcadia_view, "_visible_window_handles", return_value={99}),
                patch.object(arcadia_view, "launch_mesh_viewer", return_value=painter),
                patch.object(arcadia_view, "launch_graph", return_value=graph) as launch_graph,
                patch.object(arcadia_view, "launch_chat", return_value=chat) as launch_chat,
                patch.object(arcadia_view, "_move_process_window", return_value=True) as move,
                patch.object(arcadia_view.time, "sleep"),
                patch.object(arcadia_view, "_close_process") as close_process,
            ):
                result = arcadia_view.run_workspace(mesh, db, chat=True)

            self.assertEqual(result, 0)
            move.assert_called_once_with(
                painter,
                (0, 0, 600, 800),
                title="Heatmap Painter - mesh.ply",
                excluded_handles={99},
            )
            launch_graph.assert_called_once_with(db.resolve(), (600, 0, 600, 400))
            launch_chat.assert_called_once_with(db.resolve(), (600, 400, 600, 400))
            self.assertEqual(
                close_process.call_args_list,
                [call(graph), call(chat), call(painter)],
            )

    def test_chat_reloads_database_before_each_question(self):
        initial = {"nodes": [], "spatial_edges": []}
        updated = {"nodes": [{"id": "house_0"}], "spatial_edges": []}
        client = object()

        with (
            patch.object(arcadia_view, "load_dotenv"),
            patch.object(arcadia_view, "OpenAI", return_value=client),
            patch.object(
                arcadia_view,
                "load_scene_db",
                side_effect=[initial, updated],
            ) as load_db,
            patch("builtins.input", side_effect=["What changed?", "exit"]),
            patch.object(arcadia_view, "ask_scene", return_value="Updated") as ask,
            patch("builtins.print"),
        ):
            arcadia_view.run_chat("graph.db")

        self.assertEqual(load_db.call_count, 2)
        ask.assert_called_once_with(client, updated, "What changed?")


if __name__ == "__main__":
    unittest.main()
