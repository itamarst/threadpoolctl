from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from threading import Thread
from typing import Callable

import numpy as np
import threadpoolctl
import psutil

try:
    from tests._openmp_test_helper.openmp_helpers_inner import check_openmp_num_threads
except ImportError:
    check_openmp_num_threads = None


def start_counting_threads() -> Callable[[], int]:
    num_threads = 0
    stop = False

    def in_thread():
        nonlocal num_threads, stop
        process = psutil.Process()
        while not stop:
            num_threads = max(num_threads, process.num_threads())

    thread = Thread(target=in_thread)
    thread.start()

    def finish():
        nonlocal stop, num_threads
        stop = True
        thread.join()
        return num_threads

    return finish


def blas():
    A = np.ones((10_000_000,))

    def blas_math(_):
        A.dot(A)

    with ThreadPoolExecutor(100) as pool:
        for _ in pool.map(blas_math, range(400)):
            pass


def openmp():
    if check_openmp_num_threads is None:
        print("OpenMP not available")
        return

    def run_openmp(_):
        check_openmp_num_threads(1000)

    with ThreadPoolExecutor(100) as pool:
        for _ in pool.map(run_openmp, range(400)):
            pass


def run(which: str) -> None:
    if which == "blas":
        func = blas
    else:
        func = openmp

    count_threads = start_counting_threads()
    func()
    max_num_threads = count_threads()

    # We're creating 100 Python threads, and asking for 2 native threads. If
    # number is close to 100, we'll assume process-wide shared thread pool. If
    # the number is close to 200, we'll assume a thread pool per thread.
    if max_num_threads < 130:
        scope = "process-wide shared thread pool"
    elif max_num_threads > 170:
        scope = "thread pool per thread"
    else:
        scope = "not sure"

    print(f"== Observed behavior: {which} ==")
    print("Maximum number of observed threads:", max_num_threads)
    print("Presumed API scope:", scope)
    print()


if __name__ == "__main__":
    threadpoolctl.threadpool_limits(2)
    print("== Library info ==")
    pprint(threadpoolctl.threadpool_info())
    print()

    run("openmp")
    print()
    run("blas")
