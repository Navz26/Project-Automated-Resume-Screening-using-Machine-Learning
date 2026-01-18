
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import save_model

# Load datasets
df_resume = pd.read_csv("../data/Resume.csv")
df_train = pd.read_csv("../data/train.csv")
df_validate = pd.read_csv("../data/validate.csv")
df_test = pd.read_csv("../data/test.csv")

# Combine train + validate
df_combined = pd.concat([df_train, df_validate], ignore_index=True)

# Features and labels
texts = df_combined["resume_str"]
labels = df_combined["category"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

# Split for training/testing
X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Train, evaluate, save models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(classification_report(y_test, preds))
    filename = name.lower().replace(" ", "_") + ".pkl"
    save_model(model, filename)
    print(f"{name} saved as models/{filename}")

# Save vectorizer
save_model(vectorizer, "vectorizer.pkl")
