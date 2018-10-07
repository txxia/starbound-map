import asyncio
import dataclasses as dc
import logging
import threading
import typing as tp


@dc.dataclass(frozen=True)
class JobProgress:
    processed_count: int
    total_count: int

    @property
    def percentage(self) -> float:
        return self.processed_count / self.total_count * 100


@dc.dataclass(frozen=True)
class AsyncJobParameters:
    inputs: tp.Iterator[tp.Any]
    input_size: int
    sync_fn: tp.Callable[[int, tp.Any], None]
    name: str = ""
    batch_size: int = 0


class AsyncJobState:
    i: int = 0


class AsyncJob:
    """
    Executes a synchronous function with batched input.
    """

    def __init__(self,
                 params: AsyncJobParameters,
                 state: AsyncJobState):
        self.params = params
        self.state = state

    @property
    def progress(self) -> JobProgress:
        return JobProgress(self.state.i, self.params.input_size)


@dc.dataclass
class _Context(threading.local):
    job_queue: asyncio.Queue
    worker: asyncio.Task

    current_job: AsyncJob = None


_ctx: _Context = None


def _context() -> _Context:
    global _ctx
    return _ctx


async def _worker_coro():
    logging.debug(f"Async job runner has started")
    ctx = _context()
    while True:
        params = await ctx.job_queue.get()

        logging.debug(f"Job started: {params}")
        state = AsyncJobState()
        ctx.current_job = AsyncJob(params, state)

        # Run job
        # from utils.profiler import LineProfiler
        # with LineProfiler(sort='time'):
        for state.i in range(params.input_size):
            fn_input = next(params.inputs)
            params.sync_fn(state.i, fn_input)
            if state.i % params.batch_size == 0:
                await asyncio.sleep(0)
        # Set completion
        state.i = params.batch_size
        await asyncio.sleep(0)

        logging.debug(f"Job finished: {params}")
        ctx.current_job = None


# `async` ensures that the worker is started with the same event loop
async def start_worker():
    global _ctx
    assert _ctx is None, "Worker has already started"

    _ctx = _Context(
        job_queue=asyncio.Queue(),
        worker=asyncio.create_task(_worker_coro())
    )


def submit(params: AsyncJobParameters):
    logging.debug(f"Submitting job {params}")
    _context().job_queue.put_nowait(params)


def current_job() -> tp.Optional[AsyncJob]:
    return _context().current_job


def job_queue_size() -> int:
    return _context().job_queue.qsize()
