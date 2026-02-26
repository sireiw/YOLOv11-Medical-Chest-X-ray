---
description: Design-first collaborative brainstorming — refine ideas into approved specs before writing any code
---

# Brainstorming Ideas Into Designs

> Adapted from [obra/superpowers](https://github.com/obra/superpowers) brainstorming skill.

## Hard Gate

**Do NOT write any code, scaffold any project, or take any implementation action until you have presented a design and the user has approved it.** This applies to EVERY project regardless of perceived simplicity.

## Anti-Pattern: "This Is Too Simple To Need A Design"

Every project goes through this process. A todo list, a single-function utility, a config change — all of them. "Simple" projects are where unexamined assumptions cause the most wasted work. The design can be short (a few sentences for truly simple projects), but you MUST present it and get approval.

## Checklist

1. **Explore project context** — check files, docs, recent commits
2. **Ask clarifying questions** — one at a time, understand purpose/constraints/success criteria
3. **Propose 2-3 approaches** — with trade-offs and your recommendation
4. **Present design** — in sections scaled to complexity, get user approval after each section
5. **Write design doc** — save to `docs/plans/YYYY-MM-DD-<topic>-design.md`
6. **Transition to implementation** — use `/writing-plans` workflow to create implementation plan

## The Process

### Understanding the Idea
- Check out the current project state first (files, docs, recent commits)
- Ask questions **one at a time** to refine the idea
- Prefer multiple choice questions when possible
- Focus on: purpose, constraints, success criteria

### Exploring Approaches
- Propose 2-3 different approaches with trade-offs
- Lead with your recommended option and explain why

### Presenting the Design
- Scale each section to its complexity: a few sentences if straightforward, up to 200-300 words if nuanced
- Ask after each section whether it looks right so far
- Cover: architecture, components, data flow, error handling, testing
- Be ready to go back and clarify

## After the Design

1. Save validated design to `docs/plans/YYYY-MM-DD-<topic>-design.md`
2. Invoke `/writing-plans` to create detailed implementation plan
3. Do NOT start coding — writing-plans is the next step

## Key Principles
- **One question at a time** — Don't overwhelm with multiple questions
- **Multiple choice preferred** — Easier to answer than open-ended
- **YAGNI ruthlessly** — Remove unnecessary features from all designs
- **Explore alternatives** — Always propose 2-3 approaches before settling
- **Incremental validation** — Present design, get approval before moving on
