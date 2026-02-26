---
description: Execute an implementation plan task-by-task in batches, with review checkpoints between batches
---

# Executing Plans

> Adapted from [obra/superpowers](https://github.com/obra/superpowers) executing-plans skill.

## Overview

Load plan, review critically, execute tasks in batches, report for review between batches.

**Core principle:** Batch execution with checkpoints for user review.

## The Process

### Step 1: Load and Review Plan
1. Read the plan file
2. Review critically — identify any questions or concerns
3. If concerns: Raise them with user before starting
4. If no concerns: Create a task checklist and proceed

### Step 2: Execute Batch (Default: 3 tasks)

For each task:
1. Mark as in-progress
2. Follow each step exactly (plan has bite-sized steps)
3. Run verifications as specified
4. Mark as completed

### Step 3: Report

When batch is complete:
- Show what was implemented
- Show verification output
- Say: **"Ready for feedback."**

### Step 4: Continue

Based on feedback:
- Apply changes if needed
- Execute next batch
- Repeat until complete

### Step 5: Verify Completion

After all tasks complete:
- Run full test suite
- Verify all requirements met using `/verify-completion`
- Present summary of all changes

## When to STOP and Ask for Help

**Stop executing immediately when:**
- Hit a blocker mid-batch (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## Remember

- Review plan critically first
- Follow plan steps exactly
- Don't skip verifications
- Between batches: just report and wait
- Stop when blocked, don't guess
