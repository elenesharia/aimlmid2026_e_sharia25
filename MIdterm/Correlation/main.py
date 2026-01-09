import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

cordinates = {
    'x': [10, 8.20, 5.60, 3.10, 0.50, -1.90, -4.40, -6.90, -9.30],
    'y': [9.20, 7.70, 5.90, 3.40, 1.30, -0.50, -2.90, -5.10, -7.60]
}
df = pd.DataFrame(cordinates)
corr, p_value = pearsonr(df['x'], df['y'])

slope, intercept = np.polyfit(df['x'], df['y'], 1)
plt.plot(df['x'], slope*df['x'] + intercept, color='purple', linestyle='--', label='Correlation line')

print(f"Pearson's correlation coefficient: {corr:.3f}")
plt.scatter(df['x'], df['y'])
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Graph")
plt.legend()
plt.grid(True)
plt.show()
