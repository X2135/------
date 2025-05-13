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
import seaborn as sns

with open("intents_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract samples
labels, texts = [], []
for i in data["intents"]:
    for j in i["patterns"]:
        labels.append(i["tag"])
        texts.append(j)

df = pd.DataFrame({"text": texts, "label": labels})
print(df['label'].value_counts())

# Split train/test set
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.15, random_state=42)

# Initialize TF-IDF vectorizer
vectorizing = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

X_train_vec = vectorizing.fit_transform(X_train)
X_test_vec = vectorizing.transform(X_test)

# Initialize SVM model and train
SVM_classifier = LinearSVC()
SVM_classifier.fit(X_train_vec, y_train)

# Use CalibratedClassifierCV
classifier = CalibratedClassifierCV(SVM_classifier, cv="prefit")
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(classifier, "intent_classifier.pkl")
joblib.dump(vectorizing, "vectorizer.pkl")

print("Model and vectorizer have been saved as intent_classifier.pkl and vectorizer.pkl")

# Confusion Matrix
def plot_confusion_matrix(classifier, X_test_vec, y_test):
    y_pred = classifier.predict(X_test_vec)
    labels = classifier.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(18, 16))  
    sns.set(font_scale=1.3)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 12, "weight": "bold", "color": "black"}
    )
    plt.title('Confusion Matrix for Intent Recognition', fontsize=22, weight='bold')
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(classifier, X_test_vec, y_test)

# Classification report form
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
display_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)

fig, ax = plt.subplots(figsize=(12, len(display_df)*0.5 + 2))
ax.axis('off')
tbl = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    rowLabels=display_df.index,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 1.2)
plt.title('Classification Report', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('classification_report_table.png', dpi=300, bbox_inches='tight')
plt.show()


