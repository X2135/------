import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from recommendation import recommend_courses, show_recommendation_table 
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load intents.json file
with open("intents_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Extract samples
labels, texts = [], []
for i in data["intents"]:
    for j in i["patterns"]:
        labels.append(i["tag"])
        texts.append(j)

# 3. Create DataFrame
df = pd.DataFrame({"text": texts, "label": labels})

# 4. View sample distribution (optional)
print(df['label'].value_counts())

# 5. Split train/test set
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.15, random_state=42)

# 6. Initialize TF-IDF vectorizer (use n-gram, remove stopwords)
vectorizing = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

# 7. Text vectorization
X_train_vec = vectorizing.fit_transform(X_train)
X_test_vec = vectorizing.transform(X_test)

# 8. Initialize SVM model and train
SVM_classifier = LinearSVC()
SVM_classifier.fit(X_train_vec, y_train)

# 9. Use CalibratedClassifierCV to get probabilities
classifier = CalibratedClassifierCV(SVM_classifier, cv="prefit")
classifier.fit(X_train_vec, y_train)

# 10. Model prediction and evaluation
y_pred = classifier.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(classifier, "intent_classifier.pkl")
joblib.dump(vectorizing, "vectorizer.pkl")

print("Model and vectorizer have been saved as intent_classifier.pkl and vectorizer.pkl")

# (1) Confusion Matrix for Intent Recognition
def plot_confusion_matrix(classifier, X_test_vec, y_test):
    y_pred = classifier.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix for Intent Recognition")
    plt.show()

# Example usage (uncomment to run):
plot_confusion_matrix(classifier, X_test_vec, y_test)


