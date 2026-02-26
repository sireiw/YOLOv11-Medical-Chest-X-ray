---
description: Write detailed implementation plans with bite-sized tasks before touching code
---

# Writing Implementation Plans

> Adapted from [obra/superpowers](https://github.com/obra/superpowers) writing-plans skill.

## Overview

Write comprehensive implementation plans assuming the engineer has **zero context** for the codebase. Document everything: which files to touch, complete code, testing, exact commands, expected output. Give bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

## Save Plans To

`docs/plans/YYYY-MM-DD-<feature-name>.md`

## Plan Document Header

Every plan MUST start with:

```markdown
# [Feature Name] Implementation Plan

**Goal:** [One sentence describing what this builds]
**Architecture:** [2-3 sentences about approach]
**Tech Stack:** [Key technologies/libraries]

---
```

## Bite-Sized Task Granularity

Each step is **one action (2-5 minutes)**:

1. "Write the failing test" — step
2. "Run it to make sure it fails" — step
3. "Implement the minimal code to make the test pass" — step
4. "Run the tests and make sure they pass" — step
5. "Commit" — step

## Task Structure

````markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py:123-145`
- Test: `tests/exact/path/to/test.py`

**Step 1: Write the failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/path/test.py::test_name -v`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/path/test.py::test_name -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## Remember

- **Exact file paths** always
- **Complete code** in plan (not "add validation")
- **Exact commands** with expected output
- **DRY, YAGNI, TDD**, frequent commits
- Reference `/tdd` workflow for test-first discipline
- Reference `/executing-plans` workflow for execution

## After Writing the Plan

Offer execution: "Plan complete and saved. Ready to execute using `/executing-plans`?"
