import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datasets import load_dataset

dataset = load_dataset("imdb", split="train[:1000]")
df = pd.DataFrame(dataset)
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("report.txt", "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
