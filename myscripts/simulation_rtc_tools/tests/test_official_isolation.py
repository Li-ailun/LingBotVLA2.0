from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTECTED = {
    "RoboTwin_lingbot/script/eval_policy_client_lingbotvla.py": "2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e",
    "evaluation/lingbot_action_step2000/policy_server.py": "ff74c9726e722416c9c7a57d88f7be855e848b4258f8828108e8c64f8e38b032",
    "lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py": "2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e",
}


def test_protected_synchronous_entrypoints_are_unchanged():
    for relative, expected in PROTECTED.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected
