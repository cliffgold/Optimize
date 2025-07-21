# CuPy test
import cupy as cp
import numpy as np
import time

hourly = int(5*365.25*24)  # 5 years in hours

def test_cupy(numpy_1, numpy_2):
    start_time = time.time()
    cupy1 = cp.asarray(numpy_1)
    cupy2 = cp.asarray(numpy_2)
    for n in range(5):
        cupy2 -= cupy1[n]
    cupy_2_sum = cupy2.sum()
    duration = time.time() - start_time
    return cupy_2_sum, duration

def test_numpy(numpy_1, numpy_2):
    start_time = time.time()
    numpy3 = numpy_1 - numpy_2
    numpy_3_sum = numpy3.sum()
    duration = time.time() - start_time
    return numpy_3_sum, duration

def main():

    numpy_1 = np.array([np.random.rand(hourly), np.random.rand(hourly), np.random.rand(hourly), np.random.rand(hourly), np.random.rand(hourly)])
    numpy_2 = np.random.rand(hourly)

    cupy_sum, cupy_duration = test_cupy(numpy_1, numpy_2)
    numpy_sum, numpy_duration = test_numpy(numpy_1, numpy_2)

    print(f"CuPy sum: {cupy_sum}, Duration: {cupy_duration:.6f} seconds")
    print(f"Numpy sum: {numpy_sum}, Duration: {numpy_duration:.6f} seconds")


if __name__ == '__main__':
    main()
