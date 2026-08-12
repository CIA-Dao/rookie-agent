---
name: summarize
description: Summarize files, logs, or project information clearly.
allowed_tools:
  - list_dir
  - read_file
  - bash
---

Summarize the requested material from the current workspace.

Target:
$ARGUMENTS

Use tools when the source material is in files or command output.
Keep the summary structured, accurate, and focused on what the user asked for.
Call out uncertainty instead of guessing.
