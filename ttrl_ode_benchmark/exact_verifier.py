from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import warnings
from contextlib import contextmanager
from typing import Optional, Sequence

from ttrl_lane_emden.core import (
    Problem, RewardInfo, generated_to_candidates, ids_to_sympy, score_general_candidate,
)


class CandidateVerificationTimeout(BaseException):
    pass


@contextmanager
def candidate_time_limit(seconds: float):
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, 'setitimer'):
        yield
        return
    def _handler(signum, frame):
        raise CandidateVerificationTimeout(f"candidate verification exceeded {seconds:g}s")
    old = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def timeout_info(error: str, expression=None):
    return RewardInfo(
        reward=-25.0, valid_parse=False, exact_residual_zero=False,
        generality_rank=0, numerical_mse=1e12, expression=expression,
        residual=None, verified_general=False, error=error,
    )


def _worker_init():
    # One symbolic verification process should occupy one CPU core.  Prevent
    # NumPy / BLAS from creating another thread team inside every process.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def _worker_score_one(payload):
    """Top-level spawn-safe worker entry point.

    env and Problem are small/picklable.  Passing them with the task lets one
    process pool stay alive while the parent moves across many ODEs, avoiding
    process startup once per equation.  Workers never touch CUDA/model state.
    """
    env, problem, timeout_s, token_ids = payload
    verifier = ExactODEVerifier(env, problem, timeout_s, worker_pool=None)
    return verifier.score_ids(token_ids)


class ParallelVerifierPool:
    """Reusable spawn-based CPU process pool for exact symbolic verification.

    Two timeout layers are used:
      1) worker-local SIGALRM around each candidate;
      2) parent-process hard watchdog around each <=workers wave.

    Some SymPy / NumPy / C-extension calls may stay in native code long enough
    that Python's SIGALRM handler cannot run promptly. A single such candidate
    must never wedge the whole benchmark.
    """
    def __init__(self, workers: int, watchdog_slack_s: float = 3.0):
        self.workers = max(1, int(workers))
        self.watchdog_slack_s = float(watchdog_slack_s)
        self._ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn') if self.workers > 1 else None
        self._pool = None
        if self.workers > 1:
            self._start_pool()

    def _start_pool(self):
        if self.workers > 1:
            self._pool = self._ctx.Pool(processes=self.workers, initializer=_worker_init)

    def _restart_pool(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self._start_pool()

    def _score_wave(self, payloads, timeout_s: float):
        jobs = [self._pool.apply_async(_worker_score_one, (payload,)) for payload in payloads]
        out = [None] * len(jobs)
        pending = set(range(len(jobs)))
        hard_s = max(2.0, float(timeout_s) + self.watchdog_slack_s)
        deadline = time.monotonic() + hard_s

        while pending:
            for i in list(pending):
                job = jobs[i]
                if not job.ready():
                    continue
                try:
                    out[i] = job.get(timeout=0)
                except BaseException as e:
                    out[i] = timeout_info(f"WORKER_ERROR: {type(e).__name__}: {e}")
                pending.remove(i)

            if not pending:
                break

            if time.monotonic() >= deadline:
                for i in pending:
                    out[i] = timeout_info(
                        f"TIMEOUT: parent hard watchdog exceeded {hard_s:g}s"
                    )
                self._restart_pool()
                break

            time.sleep(0.01)

        return out

    def score_many(self, env, problem: Problem, timeout_s: float, token_id_batches):
        batches = [list(map(int, ids)) for ids in token_id_batches]
        if not batches:
            return []
        if self._pool is None:
            verifier = ExactODEVerifier(env, problem, timeout_s, worker_pool=None)
            return [verifier.score_ids(ids) for ids in batches]

        payloads = [(env, problem, float(timeout_s), ids) for ids in batches]
        results = []
        for s in range(0, len(payloads), self.workers):
            wave = payloads[s:s + self.workers]
            results.extend(self._score_wave(wave, timeout_s))
        return results

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def terminate(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.terminate()

def resolve_verifier_workers(value: int) -> int:
    """0 => auto (up to 32 logical CPUs); 1 => original serial verifier."""
    value = int(value)
    if value > 0:
        return value
    return max(1, min(32, int(os.cpu_count() or 1)))


class ExactODEVerifier:
    """Exact residual + generality verifier with per-candidate timeout.

    When worker_pool is provided, uncached candidates in each generated batch
    are verified concurrently in independent CPU processes.  Mathematical
    scoring and thresholds are otherwise unchanged.
    """
    def __init__(self, env, problem: Problem, candidate_timeout_s: float = 4.0,
                 worker_pool: Optional[ParallelVerifierPool] = None):
        self.env = env
        self.problem = problem
        self.candidate_timeout_s = float(candidate_timeout_s)
        self.worker_pool = worker_pool

    def score_ids(self, token_ids: Sequence[int]) -> RewardInfo:
        hyp = None
        try:
            with candidate_time_limit(self.candidate_timeout_s):
                hyp = ids_to_sympy(self.env, token_ids)
                return score_general_candidate(self.env, self.problem, hyp)
        except CandidateVerificationTimeout as e:
            return timeout_info(f"TIMEOUT: {e}", hyp)
        except BaseException as e:
            return timeout_info(f"{type(e).__name__}: {e}", hyp)

    def evaluate_generated(self, generated, gen_len, cache: Optional[dict] = None):
        candidates = generated_to_candidates(self.env, generated, gen_len)
        keys = [tuple(int(x) for x in ids) for ids, _ in candidates]

        if cache is None:
            ids_list = [ids for ids, _ in candidates]
            if self.worker_pool is not None:
                infos = self.worker_pool.score_many(
                    self.env, self.problem, self.candidate_timeout_s, ids_list
                )
            else:
                infos = [self.score_ids(ids) for ids in ids_list]
            return candidates, infos, 0, len(infos)

        # Preserve cache semantics while verifying all *unique uncached*
        # candidates concurrently.  Intra-batch duplicates therefore cost one
        # real verifier evaluation, just as they would after the first serial
        # cache insertion.
        pending = {}
        for key, (ids, _) in zip(keys, candidates):
            if key not in cache and key not in pending:
                pending[key] = ids

        pending_keys = list(pending.keys())
        pending_ids = [pending[k] for k in pending_keys]
        if pending_ids:
            if self.worker_pool is not None:
                scored = self.worker_pool.score_many(
                    self.env, self.problem, self.candidate_timeout_s, pending_ids
                )
            else:
                scored = [self.score_ids(ids) for ids in pending_ids]
            for key, z in zip(pending_keys, scored):
                cache[key] = z

        infos = [cache[key] for key in keys]
        misses = len(pending_keys)
        hits = len(candidates) - misses
        return candidates, infos, hits, misses


def exact_count(infos):
    return int(sum(bool(z.verified_general) for z in infos))


def timeout_count(infos):
    return int(sum(bool(z.error and str(z.error).startswith('TIMEOUT:')) for z in infos))
