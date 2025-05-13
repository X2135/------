# Design a pilot study of an educational AI chatbot integrating course recommendations and diverse services for the University of Leeds

## Project Outline

This is an intelligent course recommendation chatbot that integrates natural language processing and recommendation systems. The system can understand the intentions input by users and, based on the keywords provided by users, recommend the most suitable courses by using semantic matching and multi-dimensional scoring algorithms. The project adopts modern NLP technology and machine learning methods, providing an interactive course recommendation experience.


## Core Functions 

- **Intent Recognition**: Using TF-IDF vectorization and Support Vector Machine (SVM) classifier to recognize user intents (such as greetings, farewells, course recommendations, etc.)
- **Semantic Understanding**: Utilizing pre-trained BERT model (all-MiniLM-L6-v2) for semantic representation and matching
- **Course Recommendation**: Multi-dimensional scoring system based on semantic similarity, keyword matching, course ratings, and review count
- **Recommendation Explanation**: Providing reasoning for each recommended course (such as high semantic match, high rating, etc.)
- **Interactive Interface**: User-friendly interface based on Streamlit, supporting natural language dialogue
- **Session State Management**: Maintaining conversation history and context for a coherent user experience

## Technical Architecture

### Component Structure
- **Intent Recognition Model**: Identifies user input intentions, such as greetings, course recommendations, uncertainty, etc.
- **Recommendation Engine**: Converts user keywords into semantic vectors and matches them with course content
- **User Interface**: Interactive chat interface based on Streamlit

### Data Flow
1. User inputs questions or needs in the chat box
2. System identifies user intent (using SVM model)
3. Provides appropriate response or form based on intent
4. After user inputs keywords, the system converts keywords into semantic vectors
5. System calculates similarity between user vectors and course vectors
6. Generates recommendation results considering multiple dimensions
7. Displays recommended courses and explanations

## File Structure Description

- **streamlit_use.py**: Main application file, implementing Streamlit interface and chat logic
- **recommendation.py**: Core recommendation system logic, including semantic representation and course recommendation algorithms
- **intension_recognition.py**: Intent recognition model training script
- **intent_classifier.pkl**: Pre-trained intent classifier model
- **vectorizer.pkl**: Pre-trained TF-IDF vectorizer
- **intents_final.json**: Intent training dataset, modified based on the original dataset (https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset)
- **CourseraDataset-Clean.csv**: Coursera course dataset (https://www.kaggle.com/datasets/elvinrustam/coursera-dataset?utm_source=chatgpt.com)

## Installation Guide

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Main dependencies:
   - streamlit
   - scikit-learn
   - sentence-transformers
   - pandas
   - joblib
   - langdetect

3. **Download Dataset** (if not included in the repository)
   Ensure `CourseraDataset-Clean.csv` file is located in the project root directory

## Usage Instructions

### Launch Application
```bash
streamlit run streamlit_use.py
```

### Interact with the Chatbot
1. Enter questions in the chat box, such as "Can you recommend me a course?"
2. The system will recognize your intent and provide appropriate responses
3. For course recommendation requests, input keywords for fields you're interested in
4. Click the "Get Recommendations" button to get recommendations
5. View the list of recommended courses and reasons for recommendation

## Technical Implementation Details

### Intent Recognition Model
- Using TF-IDF to extract text features (including unigrams and bigrams)
- Using linear SVM classifier for intent classification
- CalibratedClassifierCV provides probability output

### Recommendation System
- Using BERT pre-trained model to convert course content and user input into semantic vectors
- Using cosine similarity to calculate semantic matching
- Scoring function considering multiple factors:
  - Semantic similarity (main factor)
  - Keyword matching
  - Course rating (≥4.5 gets bonus)
  - Review count (≥1000 gets bonus)
  - User preference level matching

### Interface and User Experience
- Building interactive interface using Streamlit
- Session state management ensures conversation coherence
- Form submission callback functions handle user input
- Responsive recommendation result display

## Extensions and Customization

### Adding New Intents
1. Add new intent categories and examples in `intents_final.json`
2. Run `intension_recognition.py` again to train the model

### Optimizing Recommendation Algorithm
Modify the scoring function in `recommendation.py` to adjust the weights of various factors or add new factors

### Dataset Update
Replace or update the `CourseraDataset-Clean.csv` file, maintaining the same column structure

## Developer
Li

This project is developed for academic research and educational purposes, suitable for learning and practicing computer science, natural language processing, and recommendation systems.

## License

This project is for academic and research purposes only. 