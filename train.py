import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datasets import load_dataset

# 加载IMDB数据，同时取正负样本（避免只有一类）
dataset = load_dataset("imdb", split="train[:500]+train[-500:]")  # 前500+后500条，确保有正负样本
df = pd.DataFrame(dataset)

# 打印标签分布，方便调试
print("标签分布：", df["label"].value_counts().to_dict())

X = df["text"]
y = df["label"]

# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = vectorizer.fit_transform(X)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y  # stratify=y保证划分后标签分布一致
)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# 保存模型和向量化器
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# 保存评估报告
with open("report.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write("标签分布：" + str(df["label"].value_counts().to_dict()) + "\n")
