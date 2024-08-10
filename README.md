# DAMP-python
Minimal python numba implementation of the DAMP algorithm, introduced in the KDD2022 paper.

Lu, Yue, et al. "Matrix profile XXIV: scaling time series anomaly detection to trillions of datapoints and ultra-fast arriving data streams." Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022.

# Example
```python
import numpy as np
import matplotlib.pyplot as plt
from damp import DAMP

TS = np.loadtxt('UCR_Anomaly_FullData/151_UCR_Anomaly_MesoplodonDensirostris_10000_19280_19440.txt')
w_size = 200
train_term = 10000
approximate_matrix_profile = DAMP(TS, w_size, train_term)
top_1_discord_start_point = approximate_matrix_profile.argmax()
plt.figure(figsize=(40,4))
plt.grid()
plt.plot(TS)
plt.axvspan(0,train_term, facecolor='green', alpha=0.2,label="train term")
plt.axvspan(top_1_discord_start_point,
             top_1_discord_start_point+w_size, facecolor='r', alpha=0.2,label="anomaly")
plt.legend()
```

## Requirements
- numpy
- numba
- rocket-fft
  - Required for FFT and IFFT to work in numba.njit
