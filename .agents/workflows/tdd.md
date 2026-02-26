---
description: Test-Driven Development — write failing test first, then minimal code, then refactor; no exceptions
---

# Test-Driven Development (TDD)

> Adapted from [obra/superpowers](https://github.com/obra/superpowers) test-driven-development skill.

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? **Delete it.** Start over. No exceptions.

## When to Use

**Always:** New features, bug fixes, refactoring, behavior changes.

**Exceptions (ask user first):** Throwaway prototypes, generated code, configuration files.

## Red-Green-Refactor Cycle

### 1. RED — Write Failing Test
- Write **one** minimal test showing what should happen
- One behavior per test, clear name, real code (no mocks unless unavoidable)

### 2. Verify RED — Watch It Fail
```bash
# Run the specific test
pytest tests/path/test_file.py::test_name -v
```
- Confirm test **fails** (not errors)
- Failure message is expected
- Fails because feature is missing, not because of typos

**Test passes?** You're testing existing behavior. Fix the test.

### 3. GREEN — Minimal Code
- Write the **simplest** code to pass the test
- Don't add features, refactor other code, or "improve" beyond the test
- YAGNI — You Aren't Gonna Need It

### 4. Verify GREEN — Watch It Pass
```bash
pytest tests/path/test_file.py -v
```
- Confirm test passes
- Other tests still pass
- No errors or warnings

**Test fails?** Fix code, not test. **Other tests fail?** Fix now.

### 5. REFACTOR — Clean Up
- Remove duplication, improve names, extract helpers
- Keep tests green. Don't add behavior.

### 6. Repeat
Next failing test for next feature.

## Good Tests

| Quality | Good | Bad |
|---------|------|-----|
| **Minimal** | One thing. "and" in name? Split it. | `test_validates_email_and_domain_and_whitespace` |
| **Clear** | Name describes behavior | `test_test1` |
| **Shows intent** | Demonstrates desired API | Obscures what code should do |

## Bug Fix Example

**Bug:** Empty email accepted

**RED:**
```python
def test_rejects_empty_email():
    result = submit_form(email='')
    assert result.error == 'Email required'
```

**Verify RED:** `FAIL: expected 'Email required', got None`

**GREEN:**
```python
def submit_form(email):
    if not email or not email.strip():
        return Result(error='Email required')
    # ...
```

**Verify GREEN:** `PASS`

## Red Flags — STOP and Start Over
- Code written before test
- Test passes immediately (not failing first)
- Can't explain why test failed
- Rationalizing "just this once"
- "I already manually tested it"
- "Keep as reference" (delete means delete)

## Verification Checklist
- [ ] Every new function/method has a test
- [ ] Watched each test fail before implementing
- [ ] Each test failed for expected reason
- [ ] Wrote minimal code to pass each test
- [ ] All tests pass with no warnings
- [ ] Edge cases and errors covered

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. |
| "I'll test after" | Tests passing immediately prove nothing. |
| "TDD will slow me down" | TDD is faster than debugging. |
| "Need to explore first" | Fine. Throw away exploration, start with TDD. |
| "Test hard = skip test" | Hard to test = hard to use. Simplify design. |
