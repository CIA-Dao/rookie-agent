from __future__ import annotations

from my_agent.core.bus.events import (
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from my_agent.core.command.evidence import (
    CommandEvidence,
    evidence_for_tool_call,
)
from my_agent.core.command.recovery import classify_command_failure


def test_evidence_normalizes_string_bash_command() -> None:
    evidence = evidence_for_tool_call(
        "bash", {"command": "node -e \"console.log('ok')\""}
    )

    assert evidence is not None
    assert evidence.original_command == "node -e \"console.log('ok')\""
    assert evidence.normalized_executable == "node"
    assert evidence.normalized_argv == ["node", "-e", '"console.log(\'ok\')"']


def test_evidence_preserves_list_command() -> None:
    evidence = evidence_for_tool_call("bash", {"command": ["npm", "run", "build"]})

    assert evidence is not None
    assert evidence.original_command == ["npm", "run", "build"]
    assert evidence.normalized_argv == ["npm", "run", "build"]


def test_non_bash_tools_do_not_get_command_evidence() -> None:
    assert evidence_for_tool_call("read_file", {"path": "src/App.vue"}) is None


def test_evidence_correlates_execution_context() -> None:
    evidence = evidence_for_tool_call(
        "bash",
        {"command": ["npm", "run", "build"]},
        run_id="run-1",
        step=4,
        tool_use_id="call-1",
        workspace_root="C:/workspace",
    )

    assert evidence is not None
    assert evidence.run_id == "run-1"
    assert evidence.step == 4
    assert evidence.tool_use_id == "call-1"
    assert evidence.cwd is not None
    assert evidence.platform is not None
    assert evidence.shell_mode is False


def test_command_failure_classification_is_stable() -> None:
    assert classify_command_failure("timeout", "tool timed out") == "timeout"
    assert (
        classify_command_failure("runtime_error", "'wc' is not recognized")
        == "unsupported-platform-command"
    )
    assert classify_command_failure("permission_denied", "timeout") == "permission-timeout"


def test_tool_events_keep_legacy_shape_and_accept_evidence() -> None:
    evidence = CommandEvidence(normalized_executable="npm")
    started = ToolCallStartedEvent(
        run_id="run",
        tool_use_id="call",
        tool_name="bash",
        params={"command": ["npm", "run", "build"]},
        ts="now",
        evidence=evidence,
    )
    finished = ToolCallFinishedEvent(
        run_id="run",
        tool_use_id="call",
        tool_name="bash",
        elapsed_ms=1,
        ts="now",
        evidence=evidence,
    )
    failed = ToolCallFailedEvent(
        run_id="run",
        tool_use_id="call",
        tool_name="bash",
        error_type="timeout",
        error_message="timed out",
        elapsed_ms=1,
        ts="now",
    )

    assert started.evidence == evidence
    assert finished.evidence == evidence
    assert failed.evidence is None
