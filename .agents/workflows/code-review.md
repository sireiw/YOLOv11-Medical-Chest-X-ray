---
description: Request and act on code review — mandatory after major features, before merges, between task batches
---

# Code Review

> Adapted from [obra/superpowers](https://github.com/obra/superpowers) requesting-code-review skill.

## When to Request Review

**Mandatory:**
- After completing a major feature
- Before merge to main
- After each batch in `/executing-plans`

**Optional but valuable:**
- When stuck (fresh perspective)
- Before refactoring (baseline check)
- After fixing a complex bug

## Review Checklist

Before requesting review, self-check:

### Correctness
- [ ] All tests pass
- [ ] No new warnings or errors
- [ ] Edge cases handled
- [ ] Error handling present

### Code Quality
- [ ] DRY — no duplicated logic
- [ ] YAGNI — no unnecessary features
- [ ] Clear naming (functions, variables, files)
- [ ] Functions are focused (single responsibility)

### Testing
- [ ] New code has corresponding tests
- [ ] Tests follow TDD (written before code)
- [ ] Tests are minimal and clear
- [ ] Edge cases and error paths tested

### Documentation
- [ ] Complex logic has comments explaining WHY
- [ ] Public APIs have docstrings
- [ ] README updated if needed

## How to Request

1. Summarize what was implemented
2. Reference the plan/requirements it fulfills
3. Show verification output (test results, build status)
4. List any known issues or trade-offs

## Acting on Feedback

| Severity | Action |
|----------|--------|
| **Critical** | Fix immediately, block progress |
| **Important** | Fix before proceeding |
| **Minor** | Note for later, don't block |

## Red Flags

- Never skip review because "it's simple"
- Never ignore Critical issues
- Never proceed with unfixed Important issues
- If reviewer is wrong, push back with technical reasoning
