import asyncio

from app.utils.log import logger


async def run_in_thread(func, *args, **kwargs):
    """Run a sync function in a thread, logging on client disconnect.

    ``asyncio.to_thread`` runs *func* in a worker thread. If the caller's
    coroutine is cancelled (client disconnects) the thread is **not**
    killable — it runs to completion. This wrapper logs the cancellation
    so orphan threads are observable rather than silent.
    """
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except asyncio.CancelledError:
        logger.debug("Client disconnected — background thread still running: %s", func.__name__)
        raise
