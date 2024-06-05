import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd

# Calculations were run 3 times and the results averaged
# Data is indexed by 'bond length', 'energy', 'standard error'
Data = np.array([[0.9,  -0.386, 0.004], \
                 [1,    -0.452, 0.003], \
                 [1.1,  -0.4945, 0.0027], \
                 [1.2,  -0.531, 0.004], \
                 [1.3,  -0.552, 0.003], \
                 [1.4,  -0.5686, 0.0021], \
                 [1.5,  -0.5822, 0.0016], \
                 [1.6,  -0.5931, 0.0024], \
                 [1.8,  -0.6008, 0.0019], \
                 [2,    -0.6025, 0.0008], \
                 [2.2,  -0.6000, 0.0016], \
                 [2.4,  -0.5968, 0.0016], \
                 [2.6,  -0.5907, 0.0014], \
                 [2.8,  -0.5845, 0.0010], \
                 [3,    -0.5765, 0.0012], \
                 [3.25, -0.5682, 0.0006], \
                 [3.5,  -0.5610, 0.0006], \
                 [4,    -0.5459, 0.0006], \
                 [4.5,  -0.5343, 0.0008], \
                 [5,    -0.52471, 0.00025], \
                 [6,    -0.51195, 0.00025], \
                 [7.5,  -0.50357, 0.00017], \
                 [10,   -0.5002, 0.0004], \
                 [100,  -0.5, 0.00001]])

# Literature data from Wind 1965
#df = pd.read_excel('H2+_exact.xlsx')
#exact_x = df.values[:,0]
#exact_E = df.values[:,3]

# Plotting the results
plt.errorbar(Data[:-1,0 ], Data[:-1,1], yerr=Data[:-1,2], capsize=4, fmt='x--', label='Data', color='tab:blue')
#plt.plot(exact_x, exact_E, label='Literature', color='tab:red')
plt.xlabel(r'Bond Length / $a_0$')
plt.ylabel(r'Energy / $E_H$')
plt.title(r'Energy of $H_2^+$ with Bond Length')
plt.legend()
plt.grid(True)
plt.show()
