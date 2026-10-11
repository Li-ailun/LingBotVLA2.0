import json
from pathlib import Path

from myscripts.inference_tools.sync_upstream_sources import verify_provenance


def test_provenance_manifest_verifies_copied_sources() -> None:
    module = Path(__file__).resolve().parents[1]
    manifest = module / "provenance.json"
    assert json.loads(manifest.read_text())["sources"]
    assert verify_provenance(manifest) == []
