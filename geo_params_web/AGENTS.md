# Application Instructions

The root [AGENTS.md](../AGENTS.md) applies here by inheritance.
Read these additional rules for application code, tests, and deployment.

## Persistence and Workflows

- Store persistent app data in a bind-mounted, ignored host directory.
- Persist job state and results in the mounted data directory.
- A tool flow may include human steps and automated jobs.
- Human input must not hold an automated processing slot.
- Mark flows waiting for user input without blocking queued jobs.
- Serialize jobs that compete for the same processing resource.
- Show automated progress as part of its parent tool flow.
- Make completed parameter collections read-only and open them as summaries.
- Put flow lists on tool landing pages, not on result pages.
- Show active flows from all tools in the global helper.

Mounts are defined in [docker-compose.yml](docker-compose.yml). The
[root README](../README.md#parameter-collection-workflows) explains storage and
user-facing flows. The shared slot in [execution_slots.py](libs/execution_slots.py)
is currently process-local; changing worker topology needs a separate
concurrency design decision.

## Workflow Tests

- Test persisted workflows across queued, failed, and restart states.
- Add regression tests for concurrent queues and workflow state changes.
- Coordinate concurrency tests with events or barriers, not timing sleeps.

From this directory, run the application suite:

```sh
pdm run python -m unittest discover -s tests -v
```

For focused checks, use discovery's `-p` option, for example:

```sh
pdm run python -m unittest discover -s tests -p test_execution_slots.py -v
```

See [development checks](../docs/development-checks.md) for cross-project checks.
