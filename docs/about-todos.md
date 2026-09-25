# About TODO Files

TODO files are high-level specifications for project objectives. They are meant
for human developers and coding agents.

- [`../TODO.md`](../TODO.md) covers repository-wide objectives.
- [`../geo_params_web/TODO.md`](../geo_params_web/TODO.md) covers the application.

A TODO must not track every edit, function, test, or implementation detail.
Include details only when an objective would likely be interpreted in different
ways or when an implementation constraint must be preserved.

## Required Structure

Each TODO uses the following sections in this order.

### Implementing

Contains the small set of objectives being worked on immediately. It normally
has no subsections. In rare cases, it may retain one subsection moved from
another state when that structure preserves useful context.

### To Implement Soon

Contains committed objectives with deadlines. Related objectives may be grouped
as a task. Every task title must include its deadline in this format:

```text
### YYYY-MM-DD - Task name
```

Place the nearest deadline first and broader deadlines farther down. Never add
an artificial deadline merely to move an idea into this section.

### Ideas

Contains possibilities without commitments or deadlines. An idea may remain in
this section after validation or completion when its main value is to document
a decision or possible direction.

### Done

Archives delivered, canceled, abandoned, or overdue objectives. Order entries
chronologically by their completion or closure date. Update each task title to
include that date:

```text
### YYYY-MM-DD - Task name
```

## Item States

Use these markers:

```text
- [ ] not completed
- [x] completed
- [-] canceled, abandoned, or overdue
```

Items in any section may use any state, including a completed item under Ideas.
Subitems may record rationale, deadlines, links, and constraints needed to
understand the objective.

## Level of Detail

- Describe desired outcomes, not a daily implementation sequence.
- Avoid lists of expected files, functions, or commits unless necessary.
- Add detail when plausible choices would produce meaningfully different results.
- Fix a technical approach only when it is an important constraint.
- Move objectives between sections without losing relevant rationale.
- Archive closed items instead of deleting them when they preserve useful context.
