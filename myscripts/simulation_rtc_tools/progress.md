# SDD ledger — plan: myscripts/simulation_rtc_tools/IMPLEMENTATION_PLAN.md

Setup ruling: workspace root is not a Git repository — worktree, per-task commits, and Git-based SDD helper scripts are unavailable — use protected-file SHA-256 plus per-task RED/GREEN test evidence instead; cost if wrong: no Git-native rollback/audit trail for this tool.
Pre-flight: Task 1 config/layout feeds Tasks 2, 3, 5, 6, and 7; names and immutable dataclasses match the plan.
Pre-flight: Task 2 protocol feeds Tasks 3, 4, 5, and 6; request/response IDs and `[50,D]` contract match the plan.
Pre-flight: Task 3 scheduler/blender feeds Task 6; promotion semantics and action-boundary contract match the spec.
Pre-flight: Task 4 worker feeds Task 6; a transport factory keeps websocket ownership isolated and testable.
Pre-flight: Task 5 server and Task 6 client share protocol payloads; both use protocol version `lingbot-robotwin-rtc/1`.
Pre-flight: Task 7 launchers consume Tasks 5 and 6 using unique RTC paths; protected synchronous paths remain untouched.

Task 1 ruling: apply_patch update operations fail with a mountinfo sandbox error — use narrowly scoped mechanical replacements only for already-created files; cost if wrong: reduced patch-level audit clarity.
Task 1: complete (tests: test_config.py → 9 passed)
Task 2: complete (tests: test_protocol.py → 8 passed)
Task 3: complete (tests: scheduler+blender → 7 passed)
Task 4: complete (tests: test_async_worker.py → 4 passed)
Task 5: complete (tests: test_policy_server_contract.py → 3 passed)
Task 6: complete (tests: eval_loop+isolation → 4 passed; copied client py_compile passed)
Task 7: complete (tests: test_entrypoints.py → 4 passed; deploy helper isolated under script/deploy_rtc)
Task 8: complete (new suite 42 passed; existing regressions 23 passed; preflight passed; real lift_pot episode completed 400 actions; trace 8 requests/8 promotions)
Final review: self-review (subagent dispatch prohibited for this session); no Critical, Important, or deferred Minor findings.
Final ruling: first RTC compile may exceed 30s — request_timeout_s is 120s and remains failure detection only, never action alignment; cost if wrong: a severely overloaded server may take up to 120s to be declared failed.
