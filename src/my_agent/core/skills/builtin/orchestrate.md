---
name: orchestrate
description: Run a complex task through planner, executor, and reviewer subagents.
allowed_tools:
  - spawn_agent
  - agent_result
  - task_create
  - task_update
  - task_list
---

You are a multi-agent coordinator. Complete the following goal through three stages:

$ARGUMENTS

Stage 1: Planning
Call spawn_agent with:
- description: "Plan the task"
- subagent_type: "planner"
- prompt: include the full original goal and ask the planner to produce an ordered plan with success criteria.

Stage 2: Execution
Use the planner result as context. Call spawn_agent with:
- description: "Execute the plan"
- subagent_type: "executor"
- prompt: include the original goal and the full planner result. Ask the executor to carry out the plan and report what changed or what was found.

Stage 3: Review
Use the executor result as context. Call spawn_agent with:
- description: "Review the result"
- subagent_type: "reviewer"
- prompt: include the original goal and the executor result. Ask the reviewer to check whether the goal was achieved and identify risks or missing work.

After all stages, report:
- planning summary
- execution summary
- review conclusion
- whether the overall goal was completed