**Task 1**
Pearson Correlation Analysis Report

The dataset consists of two variables, x and y, corresponding to the coordinates of the dots shown in the graph.
The given data was analyzed using Python code with the help of the libraries such as pandas, scipy, and matplotlib. First, the values of x and y were stored in a DataFrame. Pearson’s correlation coefficient was then calculated using the pearsonr() function. This coefficient measures the strength and direction of the linear relationship between the two variables. Finally, a scatter plot was created to visualize the relationship.
The calculated Pearson correlation coefficient is 0.999, which indicates an extremely strong positive linear correlation between the variables.
In conclusion both the numerical value of Pearson’s correlation coefficient and the graphical visualization confirm a very strong positive linear relationship between the two variables.

**Code**
""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

cordinates = {
    'x': [10, 8.20, 5.60, 3.10, 0.50, -1.90, -4.40, -6.90, -9.30],
    'y': [9.20, 7.70, 5.90, 3.40, 1.30, -0.50, -2.90, -5.10, -7.60]
}
df = pd.DataFrame(cordinates)
corr, p_value = pearsonr(df['x'], df['y'])

print(f"Pearson's correlation coefficient: {corr:.3f}")
plt.scatter(df['x'], df['y'])
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Graph")
plt.grid(True)
plt.show()
""""""""""""""""""""""""""""""""""""""""

**Graph**
<img width="806" height="721" alt="image" src="https://github.com/user-attachments/assets/686fe54b-18bb-4767-86c0-9063f662113a" />

