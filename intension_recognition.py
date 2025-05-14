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
import numpy as np

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
    # Only retain the main categories
    main_labels = [
        "recommend_course", "uncertain_feeling", "greeting",
        "goodbye", "name", "other_inquiry"
    ]
    y_pred = classifier.predict(X_test_vec)

    label_indices = [i for i, l in enumerate(classifier.classes_) if l in main_labels]
    filtered_labels = [classifier.classes_[i] for i in label_indices]

    y_test_main = [y for y in y_test if y in main_labels]
    y_pred_main = [y for i, y in enumerate(y_pred) if y_test.iloc[i] in main_labels]

    cm = confusion_matrix(y_test_main, y_pred_main, labels=main_labels)

    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=main_labels,
        yticklabels=main_labels,
        cbar=False,
        linewidths=0.3,
        linecolor='gray',
        annot_kws={"size": 12, "weight": "bold", "color": "black"}
    )
    for t in ax.texts:
        if t.get_text() == '0':
            t.set_text('')
    plt.title('Confusion Matrix (Main Intents)', fontsize=15, weight='bold', pad=10)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout(pad=0.5)
    plt.savefig('confusion_matrix_main.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(classifier, X_test_vec, y_test)

# Classification report form
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
display_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)

fig, ax = plt.subplots(figsize=(6.5, min(0.32 * len(display_df) + 1.2, 10)))
ax.axis('off')

# Beautify table: white background, bold headers, grid lines, compact columns
colors = [["#f8f8f8"]*len(display_df.columns) for _ in range(len(display_df))]
tbl = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    rowLabels=display_df.index,
    cellLoc='center',
    loc='center',
    cellColours=colors
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(0.95, 0.95)

# Bold header and compact
for key, cell in tbl.get_celld().items():
    row, col = key
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='#222')
    if row == 0:
        cell.set_facecolor('#e0e0e0')
    if col == -1:
        cell.set_facecolor('#e0e0e0')
    cell.set_linewidth(0.6)
    cell.set_edgecolor('#888')
    cell.PAD = 0.01

plt.title('Intent Classification Report', fontsize=14, weight='bold', pad=10)
plt.tight_layout(pad=0.2)
plt.savefig('classification_report_table.png', dpi=300, bbox_inches='tight')
plt.show()


