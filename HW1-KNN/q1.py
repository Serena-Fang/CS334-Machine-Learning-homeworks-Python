import numpy as np
import numpy.testing as npt
import time


def gen_random_samples():
    return np.random.randn(5000000)

def sum_squares_for(samples):
    timeElapse = 0
    ss = 0
    start_time = time.time()
    for x in samples:
        ss += x*x
    timeElapse = time.time() - start_time

    return ss, timeElapse


def sum_squares_np(samples):
    timeElapse = 0
    ss = 0
    array1 = samples
    array2 = samples
    # set the start time
    start_time = time.time()
    # apply numpy dot function
    ss = np.dot(array1,array2)
    # calculate time elapsed
    timeElapse = time.time() - start_time

    return ss, timeElapse


def main():
    # generate the random samples
    samples = gen_random_samples()
    # call the sum of squares
    ssFor, timeFor = sum_squares_for(samples)
    # call the numpy version
    ssNp, timeNp = sum_squares_np(samples)
    # make sure they're the same value
    npt.assert_almost_equal(ssFor, ssNp, decimal=5)
    # print out the values
    print("Time [sec] (for loop):", timeFor)
    print("Time [sec] (np loop):", timeNp)

if __name__ == "__main__":
    main()

# (b) It takes the for loop 0.7325472831726074 seconds to compute the squares sum.
# (c) It takes the numpy dot method 0.0014379024505615234 seconds to compute the squares sum.
# (d) The vectorized approach is about 900 times faster than the first approach.