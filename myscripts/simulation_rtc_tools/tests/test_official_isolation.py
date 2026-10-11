from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTECTED = {
    "RoboTwin_lingbot/script/eval_policy_client_lingbotvla.py": "2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e",
    "evaluation/lingbot_action_step2000/policy_server.py": "f48774adea86c189ce42c97c733da7d672ef500792f993ba6d750f31ab394662",
    "evaluation/lingbot_action_step2000/run_eval.sh": "441e1b06c690b7414306ee7b6badcccbf024bf378dd577555fe6f8db1d4ea01f",
}


def test_protected_synchronous_entrypoints_are_unchanged():
    for relative, expected in PROTECTED.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected
