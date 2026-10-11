from pathlib import Path

from myscripts.inference_tools.sync_upstream_sources import verify_provenance


def test_official_policy_snapshot_is_exact():
    manifest = Path(__file__).resolve().parents[1] / "provenance_policy.json"
    assert verify_provenance(manifest) == []
