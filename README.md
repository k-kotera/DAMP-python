# DAMP-python
Minimal python numba implementation of the DAMP algorithm, introduced in the KDD2022 paper.

Lu, Yue, et al. "Matrix profile XXIV: scaling time series anomaly detection to trillions of datapoints and ultra-fast arriving data streams." Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022.

## Requirements
- numpy
- numba
- rocket-fft
  - Required for FFT and IFFT to work in numba.njit
