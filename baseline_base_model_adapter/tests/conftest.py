# The realtime pacer has a direct smoke test in the RoboTwin environment,
# because importing RoboTwin's package initializer requires SAPIEN.
collect_ignore = ["test_realtime_pacer.py"]
