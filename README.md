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

1. Data File - https://github.com/elenesharia/aimlmid2026_e_sharia25/blob/main/MIdterm/Spam%20filter/e_sharia25_37142.csv
2. Code - https://github.com/elenesharia/aimlmid2026_e_sharia25/blob/main/MIdterm/Spam%20filter/filter.py
The dataset was imported using the pandas library from a CSV file that contains precomputed numerical features extracted from email messages. Data includes the total number of words, the number of embedded links, the count of fully capitalized words, and the number of words associated with spam-related terms. These attributes were grouped into the feature matrix X, while the target variable is_spam was stored in the label vector y (0 represents legitimate emails and 1 represents spa). This data preparation step enables the application of supervised machine learning techniques.

For classification, a logistic regression model was employed to differentiate between spam and legitimate emails. The model was implemented using the LogisticRegression class from the scikit-learn library and trained on 70% of the available dataset. The maximum number of iterations was set to 1000 to ensure fully convergence of the training process.

Logistic Regression Coefficients:
           Feature  Coefficient
           words   -   0.007250
            links   -   0.747564
    capital_words   -  0.421564
  spam_word_count   -  0.714043

3. Accuracy: 0.9613333333333334

Confusion Matrix:
 [[355  13]
 [ 16 366]]

To check how well the model works, it was tested on the part of the dataset that wasn’t used for training (30%). The logistic regression model was run on this test data to predict whether each email was spam or not. Then accuracy was measured by comparing these predictions with the actual labels, showing how often the model got it right. Additionally, a confusion matrix was generated to analyze the classification results in more detail. The confusion matrix shows the number of correctly and incorrectly classified spam and legitimate emails, allowing identification of false positives and false negatives. 

4. The code classifies email text by extracting features such as word count, hyperlinks, capitalized words, and spam-related terms. Hyperlinks are detected with regular expressions, while string analysis identifies capitalized and spam-indicative words. The resulting feature vector is evaluated by the trained logistic regression model to determine whether the email is spam or legitimate, ensuring consistency with the training process.

5. **"""ATTENTION!!!
You can EARN money NOW with this LIMITED and EXCLUSIVE offer.
WIN free money by opening the CONFIDENTIAL file attached to this email.
Click the link below to VIEW and OPEN the file:
http://freemoney.now.com
THIS private and SECRET opportunity is available for a short time only.
Hurry! Start your FREE trial now and receive diet pills as a bonus."""**

 This email was deliberately created to contain multiple characteristics commonly associated with spam messages. It includes a high number of spam-indicative words such as “earn,” “win,” “money,” “free,” “confidential,” “secret,” “exclusive,” “offer,” “click,” “hurry,” "pills", "diet" and “now,” which significantly increases the spam_word_count feature.
The message also contains a hyperlink, increasing the links feature, and uses several fully capitalized words to attract attention, raising the capital_words count.

6.  **"Dear reader,
Please find attached the project report for our course.
Let me know if any changes are needed.
Best regards,
Head of the deperatament."**

This email was intentionally written to represent a legit communication in an academic context. It uses a formal and professional tone, does not contain promotional or urgent language, and includes no spam-indicative keywords, links or capitilized words.Thus it decreases the chances of being misclassified as spam and increases the likelihood of being correctly recognized as a valid message.


7. A: Class Distribution Study - The bar chart shows the distribution of email classes in the dataset. Legitimate emails (0) and spam emails (1) are displayed side by side, allowing easy comparison. From this chart, it is possible to see whether the dataset is balanced or skewed toward one class, which is important for training a reliable classifier.

<img width="640" height="480" alt="fil1" src="https://github.com/user-attachments/assets/9de650c9-151a-43e3-b9f7-b9424f38a883" />


   B: Confusion Matrix Heatmap - The confusion matrix heatmap provides a visual representation of the model’s classification performance on the test dataset. The diagonal elements show correctly classified emails (true positives and true negatives), while the off-diagonal elements represent misclassifications (false positives and false negatives).
<img width="640" height="480" alt="fil2" src="https://github.com/user-attachments/assets/bbcb1dcc-4bae-4ecd-8625-f832e6582be6" />

