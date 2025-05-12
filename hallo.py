import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from recommendation import recommend_courses, show_recommendation_table 
import joblib

# ✅ 1. 加载 intents.json 文件
with open("intents_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ 2. 提取样本
labels, texts = [], []
for i in data["intents"]:
    for j in i["patterns"]:
        labels.append(i["tag"])
        texts.append(j)

# ✅ 3. 创建 DataFrame
df = pd.DataFrame({"text": texts, "label": labels})


# ✅ 4. 查看样本数量分布（选做）
print(df['label'].value_counts())

# ✅ 5. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.15, random_state=42)

# ✅ 6. 初始化 TF-IDF 向量器（使用 n-gram，去除停用词）
vectorizing = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

# ✅ 7. 文本向量化
X_train_vec = vectorizing.fit_transform(X_train)
X_test_vec = vectorizing.transform(X_test)

# ✅ 8. 初始化 SVM 模型并训练
SVM_classifier = LinearSVC()
SVM_classifier.fit(X_train_vec, y_train)

# ✅ 9. 使用 CalibratedClassifierCV 包装模型以获取概率
classifier = CalibratedClassifierCV(SVM_classifier, cv="prefit")
classifier.fit(X_train_vec, y_train)

# ✅ 10. 模型预测并评估
y_pred = classifier.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# ✅ 模型训练完毕后，添加保存模型与向量器
joblib.dump(classifier, "intent_classifier.pkl")
joblib.dump(vectorizing, "vectorizer.pkl")

print("✅ 模型与向量器已保存为 intent_classifier.pkl 和 vectorizer.pkl")
