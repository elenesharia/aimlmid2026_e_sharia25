**Task 1 - Finding the correlation**
Pearson Correlation Analysis Report

The dataset consists of two variables, x and y, corresponding to the coordinates of the dots shown in the graph.
The given data was analyzed using Python code with the help of the libraries such as pandas, scipy, and matplotlib. First, the values of x and y were stored in a DataFrame. Pearson’s correlation coefficient was then calculated using the pearsonr() function. This coefficient measures the strength and direction of the linear relationship between the two variables. Finally, a scatter plot was created to visualize the relationship.
The calculated Pearson correlation coefficient is**0.999**, which indicates an extremely strong positive linear correlation between the variables.
In conclusion both the numerical value of Pearson’s correlation coefficient and the graphical visualization confirm a very strong positive linear relationship between the two variables.

**Code**  (Sorry for the way code is presented, it just keeps changing formats)

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


**Graph**


<img width="640" height="480" alt="corelation1" src="https://github.com/user-attachments/assets/b28fb6c6-6512-4286-a989-d42848c06f05" />


**Task 2 - Spam email detection**

1. Data File -
2. Code -
The dataset was imported using the pandas library from a CSV file that contains precomputed numerical features extracted from email messages. Data includes the total number of words, the number of embedded links, the count of fully capitalized words, and the number of words associated with spam-related terms. These attributes were grouped into the feature matrix X, while the target variable is_spam was stored in the label vector y (0 represents legitimate emails and 1 represents spa). This data preparation step enables the application of supervised machine learning techniques.

For classification, a logistic regression model was employed to differentiate between spam and legitimate emails. The model was implemented using the LogisticRegression class from the scikit-learn library and trained on 70% of the available dataset. The maximum number of iterations was set to 1000 to ensure fully convergence of the training process.

Logistic Regression Coefficients:
           Feature  Coefficient
0            words     0.007250
1            links     0.747564
2    capital_words     0.421564
3  spam_word_count     0.714043

3. Accuracy: 0.9613333333333334

Confusion Matrix:
 [[355  13]
 [ 16 366]]

To check how well the model works, it was tested on the part of the dataset that wasn’t used for training (30%). The logistic regression model was run on this test data to predict whether each email was spam or not. Then accuracy was measured by comparing these predictions with the actual labels, showing how often the model got it right. Additionally, a confusion matrix was generated to analyze the classification results in more detail. The confusion matrix shows the number of correctly and incorrectly classified spam and legitimate emails, allowing identification of false positives and false negatives. 

