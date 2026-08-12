from __future__ import annotations


def classify_command_failure(error_type: str | None, message: str) -> str | None:
    """Map heterogeneous local-process errors to stable recovery categories."""
    lowered = message.lower()
    if error_type == "permission_denied" and "timeout" in lowered:
        return "permission-timeout"
    if error_type == "timeout" or "timed out" in lowered:
        return "timeout"
    if error_type == "process_start_error":
        if "not recognized" in lowered or "not found" in lowered:
            return "command-not-found"
        return "process-start-failure"
    if "not recognized as an internal or external command" in lowered:
        return "command-not-found"
    if "command not found" in lowered or "no such file or directory" in lowered:
        return "command-not-found"
    if any(name in lowered for name in ("findstr", "wc", "find")) and (
        "not recognized" in lowered or "not found" in lowered
    ):
        return "unsupported-platform-command"
    if "missing script" in lowered:
        return "missing-project-script"
    if error_type == "path_error":
        return "rejected"
    if error_type is not None:
        return "non-zero-exit" if error_type == "runtime_error" else error_type
    return None
