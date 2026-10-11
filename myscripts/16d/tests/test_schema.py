from __future__ import annotations

import importlib

import numpy as np
import pytest


schema = importlib.import_module("myscripts.16d.common.schema")


def test_schema_is_left_first_and_exactly_16d() -> None:
    assert schema.SCHEMA_NAME == "r1pro_dual_arm_16d_left_first_v1"
    assert schema.STATE_NAMES == (
        "left_arm_j0", "left_arm_j1", "left_arm_j2", "left_arm_j3",
        "left_arm_j4", "left_arm_j5", "left_arm_j6", "left_gripper",
        "right_arm_j0", "right_arm_j1", "right_arm_j2", "right_arm_j3",
        "right_arm_j4", "right_arm_j5", "right_arm_j6", "right_gripper",
    )
    assert schema.ACTION_NAMES == tuple(f"target_{name}" for name in schema.STATE_NAMES)


def test_groups_cover_each_dimension_once() -> None:
    flattened = [index for indices in schema.GROUPS.values() for index in indices]
    assert flattened == list(range(16))


def test_core_topics_exclude_torso() -> None:
    topics = [schema.TOPICS[key] for key in schema.CORE_TOPIC_KEYS]
    assert len(topics) == 11
    assert not any("torso" in topic.lower() for topic in topics)


def test_assembly_preserves_left_first_order() -> None:
    state = schema.assemble_state(
        left_arm=np.arange(1, 8), left_gripper=np.array([8]),
        right_arm=np.arange(9, 16), right_gripper=np.array([16]),
    )
    action = schema.assemble_action(
        left_arm=np.arange(101, 108), left_gripper=np.array([108]),
        right_arm=np.arange(109, 116), right_gripper=np.array([116]),
    )
    np.testing.assert_array_equal(state, np.arange(1, 17, dtype=np.float32))
    np.testing.assert_array_equal(action, np.arange(101, 117, dtype=np.float32))


def test_validate_vector_array_rejects_legacy_20d() -> None:
    with pytest.raises(ValueError, match=r"expected.*16.*got.*20"):
        schema.validate_vector_array(np.zeros((3, 20)), name="state", episode="episode_000001")


def test_assembly_rejects_wrong_component_size() -> None:
    with pytest.raises(ValueError, match=r"left_arm.*expected 7.*got 6"):
        schema.assemble_state(
            left_arm=np.zeros(6), left_gripper=np.zeros(1),
            right_arm=np.zeros(7), right_gripper=np.zeros(1),
        )
