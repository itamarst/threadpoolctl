import sys
from concurrent.futures import ThreadPoolExecutor
from json import dumps
from threading import Thread
from typing import Callable

import numpy as np
import threadpoolctl
import psutil

from tests._openmp_test_helper.openmp_helpers_inner import check_openmp_num_threads

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
        for _ in range(5):
            A.dot(A)

    with ThreadPoolExecutor(10) as pool:
        for _ in pool.map(blas_math, range(100)):
            pass


def openmp():
    def run_openmp(_):
        check_openmp_num_threads(1000)

    with ThreadPoolExecutor(10) as pool:
        for _ in pool.map(run_openmp, range(100)):
            pass



def run(which: str) -> None:
    print(threadpoolctl.threadpool_info())
    threadpoolctl.threadpool_limits(12)

    if which == "blas":
        func = blas
    else:
        func = openmp

    count_threads = start_counting_threads()
    func()
    max_num_threads = count_threads()
    print(dumps({"max_threads": max_num_threads}))


if __name__ == '__main__':
    run(sys.argv[1])
