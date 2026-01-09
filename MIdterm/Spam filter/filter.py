import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("e_sharia25_37142.csv")
X = df[['words', 'links', 'capital_words', 'spam_word_count']]
y = df['is_spam']

plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Spam vs Legitimate")
plt.xlabel("Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Number of Emails")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nLogistic Regression Coefficients:")
print(coef)

# ==============================
# MODEL VALIDATION
# ==============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", cm)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Spam"],
    yticklabels=["Legitimate", "Spam"]
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Heatmap")
plt.show()

# List of common spam words
spam_words = [
    "earn", "win", "money", "confidential", "secret", "free", "you", "now", "attachement", "view", "file", "open",
    "limited", "offer", "Exclusive", "now", "Hurry", "Trial", "diet", "pills", "Private", "attention", "link",
    "Click"
]

def extract_features(email_text):

    words = email_text.split()
    word_count = len(words)

    link_count = len(re.findall(r"http[s]?://", email_text))
    capital_words = sum(1 for w in words if w.isupper())
    spam_word_count = sum(
        email_text.lower().count(sw) for sw in spam_words
    )

    return [[word_count, link_count, capital_words, spam_word_count]]

def classify_email(email_text):

    features = extract_features(email_text)
    predict = model.predict(features)[0]
    prob = model.predict_proba(features)[0][predict]

    return predict, prob


spam_email = """
ATTENTION!!!
You can EARN money NOW with this LIMITED and EXCLUSIVE offer.
WIN free money by opening the CONFIDENTIAL file attached to this email.
Click the link below to VIEW and OPEN the file:
http://freemoney.now.com
THIS private and SECRET opportunity is available for a short time only.
Hurry! Start your FREE trial now and receive diet pills as a bonus.
"""

legit_email = """
Dear reader,
Please find attached the project report for our course.
Let me know if any changes are needed.
Best regards,
Head of the deperatament.
"""

label, confidence = classify_email(spam_email)
print("\nSpam Email Prediction:")
print("Class:", "Spam" if label == 1 else "Legitimate")
print("Confidence:", round(confidence, 3))

label, confidence = classify_email(legit_email)
print("\nLegitimate Email Prediction:")
print("Class:", "Spam" if label == 1 else "Legitimate")
print("Confidence:", round(confidence, 3))
