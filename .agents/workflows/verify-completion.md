---
description: Evidence-before-claims verification — never claim success without running and reading verification output
---

# Verification Before Completion

> Adapted from [obra/superpowers](https://github.com/obra/superpowers) verification-before-completion skill.

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command **in this session**, you cannot claim it passes.

## The Gate Function

Before claiming ANY status or expressing satisfaction:

1. **IDENTIFY:** What command proves this claim?
2. **RUN:** Execute the FULL command (fresh, complete)
3. **READ:** Full output, check exit code, count failures
4. **VERIFY:** Does output confirm the claim?
   - **NO →** State actual status with evidence
   - **YES →** State claim WITH evidence
5. **ONLY THEN:** Make the claim

**Skip any step = not verifying.**

## Common Claims and Requirements

| Claim | Requires | NOT Sufficient |
|-------|----------|----------------|
| Tests pass | Test command output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build command: exit 0 | Linter passing, "logs look good" |
| Bug fixed | Test original symptom: passes | Code changed, "assumed fixed" |
| Requirements met | Line-by-line checklist verified | "Tests passing" alone |

## Red Flags — STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification ("Great!", "Perfect!", "Done!")
- About to commit/push without verification
- Relying on partial verification
- Thinking "just this once"
- **ANY wording implying success without running verification**

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence ≠ evidence |
| "Just this once" | No exceptions |
| "Linter passed" | Linter ≠ tests ≠ build |
| "Partial check is enough" | Partial proves nothing |

## Key Patterns

**Tests:**
```
✅ [Run test command] → [See: 34/34 pass] → "All tests pass"
❌ "Should pass now" / "Looks correct"
```

**Bug fix regression test (TDD Red-Green):**
```
✅ Write → Run (FAIL) → Fix → Run (PASS) → Revert fix → Run (MUST FAIL) → Restore → Run (PASS)
❌ "I've written a regression test" (without red-green cycle)
```

**Build:**
```
✅ [Run build] → [See: exit 0] → "Build passes"
❌ "Linter passed" (linter ≠ build)
```

**Requirements:**
```
✅ Re-read plan → Create checklist → Verify each item → Report gaps or completion
❌ "Tests pass, phase complete"
```

## The Bottom Line

**No shortcuts for verification.** Run the command. Read the output. THEN claim the result.

This is non-negotiable.
