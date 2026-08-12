---
name: init
description: Inspect a project and draft agent-facing repository guidance.
allowed_tools:
  - list_dir
  - read_file
  - bash
---

Inspect the current workspace and prepare a concise project initialization summary.

Target:
$ARGUMENTS

Identify the project type, important directories, package or build files, common commands,
testing approach, and any conventions future agent runs should remember.
Do not modify files unless the user explicitly asks.
