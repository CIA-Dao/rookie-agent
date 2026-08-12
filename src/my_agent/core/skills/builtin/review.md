---
name: review
description: Review code for bugs, risks, and missing tests.
allowed_tools:
  - list_dir
  - read_file
  - bash
---

You are reviewing code in the current workspace.

Target:
$ARGUMENTS

Focus on concrete bugs, behavioral regressions, security risks, and missing tests.
Use tools to inspect relevant files before making claims.
Return findings first, ordered by severity, with file references when available.
