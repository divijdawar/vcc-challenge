"""Bounded CPU-only prefetch, with ordered delivery and exception propagation."""
from queue import Empty, Full, Queue
from threading import Event, Thread

from .config import config


class PrefetchIterator:
    """Pull a lazy CPU observation iterator on one worker. Use as a context manager.

    The iterator must yield NumPy/sparse observations, never create GPU tensors.
    At most depth completed steps plus the producer's current step are retained.
    depth=0 disables the worker. Closing cancels queued work; an in-flight source
    read is allowed to finish. The source iterator is closed on its owning thread.
    """

    def __init__(self, iterable, depth=config.prefetch_steps):
        if type(depth) is not int or depth < 0:
            raise ValueError("Prefetch depth must be a nonnegative integer")
        self.source = iter(iterable)
        self.depth = depth
        self._closed = False
        self._cancel = Event()
        self._queue = Queue(maxsize=max(depth, 1))
        self._worker = None

    def _put(self, item):
        while not self._cancel.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except Full:
                pass

    def _produce(self):
        try:
            for item in self.source:
                if self._cancel.is_set():
                    break
                self._put(("value", item))
            self._put(("end", None))
        except BaseException as error:
            self._put(("error", error))
        finally:
            if hasattr(self.source, "close"):
                self.source.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self.depth == 0:
            try:
                return next(self.source)
            except BaseException:
                self.close()
                raise
        if self._worker is None:
            self._worker = Thread(target=self._produce, name="vcc-observation-prefetch", daemon=True)
            self._worker.start()
        kind, value = self._queue.get()
        if kind == "value":
            return value
        self.close()
        if kind == "error":
            raise value
        raise StopIteration

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        if self._worker is None:
            if hasattr(self.source, "close"):
                self.source.close()
        else:
            self._worker.join(timeout=1.0)
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
