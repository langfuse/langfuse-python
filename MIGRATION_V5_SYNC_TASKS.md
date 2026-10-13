# Synchronous experiment tasks in v5

This note describes the execution-context change on the v5 preparation branch;
it does not announce a released SDK version.

Blocking synchronous experiment tasks now run in worker threads instead of
blocking the experiment's event loop. They can overlap up to the experiment's
`max_concurrency`. Async tasks, including recognized decorated wrappers and
callable objects, continue to run on the experiment's event loop.

## Audit captured resources before upgrading

A synchronous task must not assume it runs on the caller's thread. A SQLite
connection created by the caller uses thread checking by default and fails when
used from a worker. The same concern applies to other resources that require a
particular thread. Even `max_concurrency=1` executes the synchronous task on a
worker and does not restore the previous execution context.

Shared task state may now be accessed concurrently. Use a client that supports
concurrent use, synchronize shared state, or create each task's resource where
it is used. Turning off SQLite's thread check alone does not establish safe
concurrent ownership.

## Create and close the resource in the synchronous task

For string dataset inputs, this task opens, uses and closes its connection on
the same worker thread:

```python
import sqlite3
from contextlib import closing


def task(*, item, **kwargs):
    with closing(sqlite3.connect("experiment.sqlite")) as connection:
        return connection.execute("SELECT ?", (item["input"],)).fetchone()[0]
```

The unit test exercises this pattern through `run_experiment` with two items
and `max_concurrency=2`.

An alternative is to use an asynchronous client and create/use its resources
on the experiment's event loop. Merely wrapping blocking work in `async def`
can keep that work on the loop and serialize it again. The loop itself is not
guaranteed to be on the original caller's thread, including notebook use.

## Inspect failed items as well as successful outputs

The existing item-error policy is unchanged: task exceptions are logged and
recorded in the trace, and failed items do not appear in `item_results`.
`run_experiment` does not re-raise each task's exception. A plain script that
captures a caller-owned SQLite connection can therefore return zero item
results without raising that SQLite exception. Check the logged failures and
returned items during migration; successful completion of the runner alone
does not prove that every task succeeded.
