"""Host-side fast paths for the training loop (no numerical effect).

Each fast path sits behind its own environment switch so it can be timed on its own.
"""
import os
import random

import torch
import torch.utils.data._utils.worker as _wk
import torch.utils.data.dataloader as _tdl


def _on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


# ----------------------------------------------------------------------------------------------------------------
# MOTUS_OPT_DS_HOOK: DeepSpeed ZeRO gradient-hook bookkeeping
# ----------------------------------------------------------------------------------------------------------------

def patch_deepspeed_hook_count() -> None:
    """DeepSpeed 0.18's ZeRO-1/2 gradient hook calls `count_used_parameters_in_backward(all_params)` once per
    parameter, and that helper walks *all* trainable parameters (a `view_as` per parameter to reach its
    AccumulateGrad node, then `_will_engine_execute_node`), so one backward costs O(n_params^2) Python work
    (~1.85 M view_as calls per iteration here). The count only feeds the hook bookkeeping (the epilogue itself runs
    from the autograd-engine callback). It is a property of the running graph task: computed at the first hook of a
    graph task, i.e. before any AccumulateGrad node of that task has run and been freed, it equals the maximum the
    uncached helper would report, which is what the bookkeeping keeps (`max_expected_hooks_seen`). So cache it per
    (graph task, parameter list)."""
    import deepspeed.runtime.zero.stage_1_and_2 as s12

    orig = s12.count_used_parameters_in_backward
    if getattr(orig, "_motus_cached", False):
        return
    cache = {}

    def cached_count(parameters):
        task = torch._C._current_graph_task_id()
        key = id(parameters)
        hit = cache.get(key)
        if task != -1 and hit is not None and hit[0] == task:
            return hit[1]
        n = orig(parameters)
        cache[key] = (task, n)
        return n

    cached_count._motus_cached = True
    s12.count_used_parameters_in_backward = cached_count


# ----------------------------------------------------------------------------------------------------------------
# MOTUS_OPT_LOADER_REUSE: keep the data-loader workers across epochs, re-seeded exactly like fresh workers
# ----------------------------------------------------------------------------------------------------------------
#
# Upstream's train loader is not persistent: every epoch (9 iterations here, because accelerate re-shards the
# DistributedSampler-sharded loader) creates a new `_MultiProcessingDataLoaderIter`, which draws a base seed from the
# main process's torch CPU stream and forks 16 new workers; worker w seeds `random`, torch and numpy from
# base_seed + w and gets the epoch's batches w, w+16, ... in order. Joining the old workers and forking 16 new ones
# from the training process costs ~4.4 s per epoch. Here the workers live on, and at every new epoch the main
# process draws the base seed exactly as a new iterator would and sends it with the `_ResumeIteration` message that
# restarts the workers; each worker then re-seeds `random`, torch and numpy (and its WorkerInfo) exactly as
# `_worker_loop` seeds a fresh worker before it touches the epoch's first index. Dispatch order (round robin from
# worker 0, prefetch) is torch's own. The batches and the main-process random stream are therefore identical.

_ORIG_WORKER_LOOP = _wk._worker_loop
_RESEED_TAG = "motus-reseed"


def _reseed_worker(base_seed: int, worker_id: int, num_workers: int, dataset) -> None:
    # exactly torch.utils.data._utils.worker._worker_loop's seeding of a fresh worker
    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    if _wk.HAS_NUMPY:
        np_seed = _wk._generate_state(base_seed, worker_id)
        import numpy as np

        np.random.seed(np_seed)
    _wk._worker_info = _wk.WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)


class _ReseedingIndexQueue:
    """Wraps a worker's index queue (inside the worker process): a tagged `_ResumeIteration` re-seeds the worker
    before torch's loop acknowledges it and recreates its fetcher."""

    def __init__(self, q, worker_id, num_workers, dataset):
        self._q, self._worker_id, self._num_workers, self._dataset = q, worker_id, num_workers, dataset

    def get(self, *args, **kwargs):
        r = self._q.get(*args, **kwargs)
        if isinstance(r, _wk._ResumeIteration):
            seed = getattr(r, "seed", None)
            if isinstance(seed, tuple) and len(seed) == 2 and seed[0] == _RESEED_TAG:
                _reseed_worker(seed[1], self._worker_id, self._num_workers, self._dataset)
                r = _wk._ResumeIteration(None)
        return r

    def __getattr__(self, name):
        return getattr(self._q, name)


def _reseeding_worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event, auto_collation, collate_fn,
                           drop_last, base_seed, init_fn, worker_id, num_workers, persistent_workers, shared_seed):
    q = _ReseedingIndexQueue(index_queue, worker_id, num_workers, dataset)
    return _ORIG_WORKER_LOOP(dataset_kind, dataset, q, data_queue, done_event, auto_collation, collate_fn, drop_last,
                             base_seed, init_fn, worker_id, num_workers, persistent_workers, shared_seed)


class _ReseedingMultiProcessingIter(_tdl._MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        # the first epoch is a plain new iterator (base seed drawn in _BaseDataLoaderIter.__init__); only the
        # workers' loop is wrapped so that later epochs can re-seed them
        _wk._worker_loop = _reseeding_worker_loop
        try:
            super().__init__(loader)
        finally:
            _wk._worker_loop = _ORIG_WORKER_LOOP
        self._persistent_workers = True     # keep the workers at the end of an epoch

    def _new_epoch(self, loader):
        # what a new iterator does: iter(sampler) (in _reset) and the base-seed draw from the loader's generator
        # (the sampler is a DistributedSampler with its own generator; it does not touch the global stream)
        base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._base_seed = base_seed
        self._shared_seed = (_RESEED_TAG, base_seed)
        try:
            self._reset(loader)
        finally:
            self._shared_seed = None


def install_reseeding_loader(dataloader) -> bool:
    """Make `dataloader` (accelerate's DataLoaderShard or a torch DataLoader) reuse its workers across epochs."""
    from torch.utils.data import IterableDataset

    base = getattr(dataloader, "base_dataloader", dataloader)
    if not isinstance(base, _tdl.DataLoader) or getattr(base, "_motus_reseeding", False):
        return False
    if base.num_workers == 0 or base.persistent_workers or isinstance(base.dataset, IterableDataset):
        return False
    state = {}

    def get_iterator():
        it = state.get("it")
        if it is None or it._shutdown:
            it = _ReseedingMultiProcessingIter(base)
            state["it"] = it
        else:
            it._new_epoch(base)
        return it

    base._get_iterator = get_iterator
    base._motus_reseeding = True
    return True


# ----------------------------------------------------------------------------------------------------------------
# MOTUS_OPT_LOADER_SPEC: persistent worker pool that also prepares the next epoch ahead of time
# ----------------------------------------------------------------------------------------------------------------
#
# Even with reused workers, an epoch start still waits ~2.3 s for its first batch (all ranks' workers start their
# 8-sample batches at once). Batch k of an epoch is fully determined by the epoch's base seed b and k: in upstream
# it is produced by worker w = k % num_workers, seeded with b + w, as that worker's (k // num_workers)-th batch.
# The dataset ignores the index (it samples task, episode, frames and instruction with `random`), so the content
# depends only on (b, k) and the batch sizes.
#
# Here a persistent pool of num_workers processes executes "slot" tasks: (b, slot w, [its batches k = w, w + n, ...])
# re-seeded exactly like a fresh worker w, any process may run any slot. At every epoch start the main process
# creates a `_BaseDataLoaderIter` (iter(sampler) + the base-seed draw from its torch CPU stream, exactly as upstream)
# and submits the epoch's slots. It then predicts the next epoch's base seed by replaying, on a copy of the CPU
# generator, the draws the training steps of this epoch make (Motus.training_step: two
# torch.randint(0, 1000, (B,)) per iteration) followed by the base-seed draw, and submits the next epoch's slots
# speculatively. When the next epoch starts, its real base seed and batch sizes are compared with the prediction:
# equal -> the prepared batches are used (identical by construction); different -> they are dropped and the epoch is
# produced normally. Either way the batches and the training random stream are those of upstream.

_SPEC_DRAWS = ((0, 1000), (0, 1000))     # per iteration, CPU stream: video and action timestep ids (B,) each


def _spec_worker_loop(dataset, collate_fn, drop_last, task_q, result_q, num_workers):
    import queue
    from torch._utils import ExceptionWrapper
    from torch.utils.data import _DatasetKind
    from torch.utils.data._utils import signal_handling

    signal_handling._set_worker_signal_handlers()
    torch.multiprocessing._set_thread_name("pt_data_worker")
    torch.set_num_threads(1)
    watchdog = _wk.ManagerWatchdog()
    while watchdog.is_alive():
        try:
            task = task_q.get(timeout=5.0)
        except queue.Empty:
            continue
        if task is None:
            break
        key, base_seed, slot, jobs = task
        _reseed_worker(base_seed, slot, num_workers, dataset)        # as _worker_loop seeds a fresh worker `slot`
        fetcher = _DatasetKind.create_fetcher(_DatasetKind.Map, dataset, True, collate_fn, drop_last)
        for k, indices in jobs:
            try:
                data = fetcher.fetch(indices)
            except Exception:
                data = ExceptionWrapper(where=f"in DataLoader worker process {slot}")
            result_q.put((key, k, data))
            del data


class _SpecPool:
    def __init__(self, loader):
        import threading
        ctx = loader.multiprocessing_context or torch.multiprocessing
        self.loader = loader
        self.nw = loader.num_workers
        self.task_q = ctx.Queue()
        self.result_q = ctx.Queue()
        self.workers = []
        for _ in range(self.nw):
            w = ctx.Process(target=_spec_worker_loop, args=(loader.dataset, loader.collate_fn, loader.drop_last,
                                                             self.task_q, self.result_q, self.nw))
            w.daemon = True
            w.start()
            self.workers.append(w)
        self.pin = bool(loader.pin_memory) and torch.cuda.is_available()
        self.device_id = torch.cuda.current_device() if self.pin else None
        self.cv = threading.Condition()
        self.store, self.discard = {}, set()
        self.next_key, self.spec = 0, None
        self.stats = {"hit": 0, "miss": 0, "epochs": 0}
        self.rank = os.environ.get("RANK", "0")
        threading.Thread(target=self._receive, name="motus_loader_pin", daemon=True).start()

    def _receive(self):
        import queue
        from torch._utils import ExceptionWrapper
        from torch.utils.data._utils.pin_memory import pin_memory
        torch.set_num_threads(1)
        if self.pin:
            torch.cuda.set_device(self.device_id)
        while True:
            try:
                key, k, data = self.result_q.get(timeout=5.0)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return
            with self.cv:
                drop = key in self.discard
            if drop:
                del data
                continue
            if self.pin and not isinstance(data, ExceptionWrapper):
                try:
                    data = pin_memory(data, None)
                except Exception:
                    data = ExceptionWrapper(where=f"in pin memory thread for device {self.device_id}")
            with self.cv:
                self.store[(key, k)] = data
                self.cv.notify_all()

    def _submit(self, base_seed, batches):
        self.next_key += 1
        key = self.next_key
        n = len(batches)
        for slot in range(min(self.nw, n)):
            self.task_q.put((key, base_seed, slot, [(k, batches[k]) for k in range(slot, n, self.nw)]))
        return key

    def start_epoch(self, base_seed, sampler_iter):
        batches = [list(b) for b in sampler_iter]
        sizes = [len(b) for b in batches]
        spec, self.spec = self.spec, None
        if spec is not None and spec[1] == base_seed and spec[2] == sizes:
            key = spec[0]
            self.stats["hit"] += 1
        else:
            if spec is not None:
                self.stats["miss"] += 1
                with self.cv:
                    self.discard.add(spec[0])
                    for kk in [kk for kk in self.store if kk[0] == spec[0]]:
                        del self.store[kk]
            key = self._submit(base_seed, batches)
        self.stats["epochs"] += 1
        if self.rank == "0" and self.stats["epochs"] <= 6:
            import sys
            print(f"[loader-spec] epoch {self.stats['epochs']}: {self.stats}", file=sys.stderr, flush=True)
        if self.loader.generator is None and batches:
            g = torch.Generator()
            g.set_state(torch.get_rng_state())
            for size in sizes:
                for low, high in _SPEC_DRAWS:
                    torch.randint(low, high, (size,), generator=g)
            pred = torch.empty((), dtype=torch.int64).random_(generator=g).item()
            # the next epoch's indices are not known yet; the dataset ignores them (see above)
            self.spec = (self._submit(pred, batches), pred, sizes)
        return key, len(batches)

    def take(self, key, k):
        with self.cv:
            while (key, k) not in self.store:
                if not self.cv.wait(timeout=5.0):
                    dead = [w.pid for w in self.workers if not w.is_alive()]
                    if dead:
                        raise RuntimeError(f"DataLoader worker(s) {dead} exited unexpectedly")
            return self.store.pop((key, k))


class _SpecEpochIter(_tdl._BaseDataLoaderIter):
    def __init__(self, loader, pool):
        super().__init__(loader)        # iter(sampler) and the base-seed draw, exactly as a new torch iterator
        self._pool = pool
        self._key, self._n = pool.start_epoch(self._base_seed, self._sampler_iter)
        self._k = 0

    def _next_data(self):
        from torch._utils import ExceptionWrapper
        if self._k >= self._n:
            raise StopIteration
        data = self._pool.take(self._key, self._k)
        self._k += 1
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data


def install_spec_loader(dataloader) -> bool:
    from torch.utils.data import IterableDataset

    base = getattr(dataloader, "base_dataloader", dataloader)
    if not isinstance(base, _tdl.DataLoader) or getattr(base, "_motus_reseeding", False):
        return False
    if base.num_workers == 0 or base.persistent_workers or isinstance(base.dataset, IterableDataset):
        return False
    if not getattr(type(base.dataset), "_motus_index_free", False) or base.batch_sampler is None:
        return False
    if getattr(base, "_dataset_kind", None) != _tdl._DatasetKind.Map:
        return False
    state = {}

    def get_iterator():
        pool = state.get("pool")
        if pool is None:
            pool = state["pool"] = _SpecPool(base)
        return _SpecEpochIter(base, pool)

    base._get_iterator = get_iterator
    base._motus_reseeding = True
    return True


# ----------------------------------------------------------------------------------------------------------------
# Diagnostics (no effect on training)
# ----------------------------------------------------------------------------------------------------------------

def install_loader_watchdog() -> None:
    """Diagnostics only: report every data-loader next() longer than 0.3 s (all ranks); on rank 0, dump the stacks
    of the main process and of its loader workers (SIGUSR1 + faulthandler, inherited through fork) for the first
    two stalls."""
    import faulthandler
    import signal
    import sys
    import threading
    import time
    rank = os.environ.get("RANK", "?")
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    orig = _tdl._MultiProcessingDataLoaderIter.__next__
    state = {"t": None, "it": None, "dumped": False, "n": 0, "calls": 0}

    def nxt(self):
        state["t"], state["it"], state["dumped"] = time.perf_counter(), self, False
        state["calls"] += 1
        try:
            return orig(self)
        finally:
            dt = time.perf_counter() - state["t"]
            state["t"] = None
            if dt > 0.3:
                print(f"[loader] rank {rank} next() #{state['calls']} waited {dt:.2f} s", file=sys.stderr, flush=True)

    _tdl._MultiProcessingDataLoaderIter.__next__ = nxt
    if rank != "0":
        return

    def watch():
        while True:
            time.sleep(0.05)
            t = state["t"]
            if t is not None and not state["dumped"] and state["n"] < 2 and time.perf_counter() - t > 1.0:
                state["dumped"] = True
                state["n"] += 1
                print("[watchdog] next() > 1 s; main process:", file=sys.stderr, flush=True)
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                for w in getattr(state["it"], "_workers", []):
                    try:
                        print(f"[watchdog] worker pid {w.pid}:", file=sys.stderr, flush=True)
                        os.kill(w.pid, signal.SIGUSR1)
                        time.sleep(0.02)
                    except Exception as e:
                        print(f"[watchdog] {e}", file=sys.stderr, flush=True)

    threading.Thread(target=watch, daemon=True).start()


# ----------------------------------------------------------------------------------------------------------------
# entry points used by train/train.py
# ----------------------------------------------------------------------------------------------------------------

def apply_all() -> None:
    """At import of train/train.py."""
    if _on("MOTUS_OPT_DS_HOOK"):
        patch_deepspeed_hook_count()
    if _on("MOTUS_DIAG_LOADER"):
        install_loader_watchdog()


def before_train_loop(trainer) -> None:
    """Called by UniDiffuserTrainer.train() right before the data iterator (and its workers) is created."""
    if _on("MOTUS_OPT_LOADER_SPEC"):
        if install_spec_loader(trainer.train_dataloader):
            return
    if _on("MOTUS_OPT_LOADER_REUSE"):
        install_reseeding_loader(trainer.train_dataloader)
