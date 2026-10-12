from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1]


def test_ros_client_has_no_rlinf_runtime_paths():
    source = (TOOLS_ROOT / "r1pro_ros_client_node.py").read_text(encoding="utf-8")

    assert "/home/lixiang/PROJECT/RLinf" not in source
