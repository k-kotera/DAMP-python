import numpy as np
from numba import njit, prange
from numpy.fft import fft, ifft

@njit
def DAMP(T, m, spIndex):
    """
    DAMP algorithm for time series discords discovery.
    
    Parameters
    T : 1D time series data 
    m : Subsequence length
    spIndex : Location of split point between training and test data
    Returns
    aMP : Left approximate Matrix Profile
    """
    aMP = np.zeros(len(T)-m+1)
    best_so_far = 0.
    PV = np.ones(len(T)-m+1)
    for i in range(spIndex, len(T)-m+1):
        if PV[i] == False:
            aMP[i] = aMP[i-1]
        else:
            aMP[i], best_so_far = DAMP_backward_processing(T,m,i,best_so_far)
            PV = DAMP_forward_processing(T,m,i,best_so_far,PV)
    return aMP

@njit
def DAMP_forward_processing(T, m, i, best_so_far, PV):
    """
    Parameters
    T : 1D time series data 
    m : Subsequence length
    i : Index of current query
    best_so_far : Highest discord score so far
    PV : Pruned Vector
    Returns
    PV : Updated PV
    """
    if i + m >= len(T) - m + 1:
        return PV
    lookahead = int(2**nextpow2(m))
    start = i + m
    end = min(start + lookahead, len(T))
    D_i = MASS_njit(T[start:end], T[i:i+m]).real
    indexes = np.where(D_i <= best_so_far)[0] + start
    PV[indexes] = 0
    return PV

@njit
def DAMP_backward_processing(T, m, i, best_so_far):
    """
    Parameters
    T : 1D time series data 
    m : Subsequence length
    i : Index of current query
    best_so_far : Highest discord score so far
    Returns
    aMPi : Discord value at position i
    best_so_far : Updated best_so_far
    """
    aMPi = np.inf
    prefix = int(2**nextpow2(m))
    while aMPi >= best_so_far:
        if i - prefix <= 0:
            aMPi = min(MASS_njit(T[:i], T[i:i+m]).real)
            if aMPi > best_so_far:
                best_so_far = aMPi
            break
        else:
            aMPi = min(MASS_njit(T[i-prefix:i], T[i:i+m]).real)
            if aMPi < best_so_far:
                break
            else:
                prefix = int(2 * prefix)
    return aMPi, best_so_far

@njit
def nextpow2(x):
    return int(np.ceil(np.log2(x)))

@njit
def MASS_njit(x, y):
    m = len(y)
    n = len(x)

    # compute y stats
    meany = np.mean(y)
    sigmay = np.std(y)

    # compute x stats
    #meanx = np.convolve(x, np.ones(m)/m, mode='valid')
    #sigmax = np.sqrt(np.convolve(x**2, np.ones(m)/m, mode='valid') - meanx**2)
    meanx,sigmax = moving_mean_sigma(x,m)
    #y = np.pad(y, (0, n - m), 'constant')  # Append zeros
    y2 = np.zeros(n)
    y2[:m] = y[::-1]

    # The main trick of getting dot products in O(n log n) time
    X = fft(x)
    Y = fft(y2)
    Z = X * Y
    z = ifft(Z).real

    dist = 2 * (m - (z[m-1:n] - m * meanx * meany) / (sigmax * sigmay))
    dist = np.sqrt(dist)
    return dist

@njit(parallel=True, fastmath=True)
def moving_mean_sigma(x, m):
    n = len(x)
    result_mean = np.empty(n - m + 1, dtype=x.dtype)
    result_std = np.empty(n - m + 1, dtype=x.dtype)
    
    window = x[:m]
    sum_x = np.sum(window)
    sum_x_sq = np.sum(window**2)
    
    result_mean[0] = sum_x / m
    result_std[0] = np.sqrt((sum_x_sq / m) - (result_mean[0]**2))
    
    for i in range(1, n - m + 1):
        sum_x += x[i+m-1] - x[i-1]
        sum_x_sq += x[i+m-1]**2 - x[i-1]**2
        
        mean = sum_x / m
        result_mean[i] = mean
        result_std[i] = np.sqrt((sum_x_sq / m) - (mean**2))
    
    return result_mean, result_std
