import math
import timeit
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from integrate import *


def integrate_threaded(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    n_iter: int = 100_000,
    n_jobs: int = 2
) -> float:
    """
    Численное интегрирование с использованием потоков (ThreadPoolExecutor).
    """
    step = (b - a) / n_jobs
    iter_per_job = n_iter // n_jobs

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                integrate,
                f,
                a + i * step,
                a + (i + 1) * step,
                n_iter=iter_per_job
            )
            for i in range(n_jobs)
        ]

        return sum(f.result() for f in as_completed(futures))


if __name__ == "__main__":
    print("Проверка корректности:")
    print(integrate(math.cos, 0, math.pi, n_iter=1_000_000))
    print(integrate_threaded(math.cos, 0, math.pi, n_iter=1_000_000, n_jobs=4))

    print("\nЗамеры времени (потоки):")
    for jobs in [1, 2, 4, 6, 8]:
        t = timeit.timeit(
            lambda: integrate_threaded(
                math.cos,
                0,
                math.pi,
                n_iter=1_000_000,
                n_jobs=jobs
            ),
            number=3
        )
        print(f"n_jobs={jobs}: {t:.5f} секунд")
