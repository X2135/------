# Design a pilot study of an educational AI chatbot integrating course recommendations and diverse services for the University of Leeds

## Project Outline

This is an intelligent course recommendation chatbot that integrates natural language processing and recommendation systems. The system can understand the intentions input by users and, based on the keywords provided by users, recommend the most suitable courses by using semantic matching and multi-dimensional scoring algorithms. The project adopts modern NLP technology and machine learning methods, providing an interactive course recommendation experience.

---
## ✨ Key Features

* **Natural-language dialogue** – chat style interface built with Streamlit.
* **Intent Recognition** – TF-IDF + Linear-SVM with probability calibration (`intent_classifier.pkl`).
* **Course Recommendation** – semantic matching using `sentence-transformers/all-MiniLM-L6-v2`, keyword boosting, rating & review heuristics.
* **Explainable Output** – every recommendation is accompanied by a human-readable reason.
* **Session Memory** – Streamlit *session_state* stores chat history, keywords and results.
* **One-click Testing** – `recommendation_test.py` generates tables / figures for your report.
* **Visual Evaluation** – confusion matrix & classification report automatically exported as PNG.

---
## 🗂️ Repository Layout

```
├── streamlit_use.py               # Web UI / dialogue loop
├── recommendation.py              # Recommendation logic
├── recommendation_test.py         # Batch test & visualisation script
├── intension_recognition.py       # Train intent model + evaluation plots
├── intent_classifier.pkl          # Trained SVM intent model
├── vectorizer.pkl                 # Trained TF-IDF vectoriser
├── CourseraDataset-Clean.csv      # Course dataset (cleaned)
├── intents_final.json             # Intent training data
├── classification_report_table.png
├── confusion_matrix_main.png
└── all_recommendations_table.png
```

---
## 🚀 Installation

```bash
# 1. clone repo
$ git clone <repo-url>
$ cd SmartCourseChatbot

# 2. create env (optional)
$ python -m venv venv && source venv/bin/activate

# 3. install deps
$ pip install -r requirements.txt
```
*First run will download the 100 MB Sentence-Transformer model automatically.*

---
## Quick Start

### Launch chatbot
```bash
streamlit run streamlit_use.py
```
Open the browser link (usually `http://localhost:8501`).  
Type something like *"Can you recommend me a data-science course?"*.

### Batch-generate evaluation figures
```bash
python recommendation_test.py      # saves PNG tables in project root
```

---
## Evaluation Metrics

Intent model (93-sample test set):
* Accuracy 0.88, macro-F1 0.88
* Confusion-matrix and per-class report exported in `/`.

Recommendation quality (manual sampling):
* >90 % courses contain at least one user keyword.
* Average rating ≥4.5 on recommended list.

---
##  Retraining Intent Model
```bash
python intension_recognition.py    # retrains & overwrites *.pkl
```
Adjust `intents_final.json` to add new intents before retraining.

---
## Customisation Tips

* **New data source** – drop another CSV with the same columns and change the path in `recommendation.py`.
* **Scoring weights** – tweak function `score()` in `recommendation.py`.
* **UI wording** – edit messages in `streamlit_use.py`.

---
## License & Citation

This repository is released for academic & educational use only.  
If you use the code or ideas, please cite the project. 