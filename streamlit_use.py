import joblib
import streamlit as st
from recommendation import recommend_courses, show_recommendation_table, generate_reason

# Load model and vectorizer
classifier = joblib.load("intent_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Intent recognition function (Top-K sorting)
def predict_top_k_intents(user_input, top_k=3):
    user_vec = vectorizer.transform([user_input])
    probs = classifier.predict_proba(user_vec)[0]
    top_indices = probs.argsort()[::-1][:top_k]
    top_intents = [(classifier.classes_[i], probs[i]) for i in top_indices]
    return top_intents

# Page settings
st.set_page_config(page_title="🎓 Smart Course Assistant", layout="centered")
st.title("🤖 Smart Course Recommendation Chatbot")

# Chat history state management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Context memory
if "context" not in st.session_state:
    st.session_state.context = {}

# Keywords and recommendations storage
if "keywords" not in st.session_state:
    st.session_state.keywords = ""
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "intent_processed" not in st.session_state:
    st.session_state.intent_processed = False
if "current_intent" not in st.session_state:
    st.session_state.current_intent = None

# Callback functions to handle form submissions
def process_uncertain_feeling_keywords():
    keyword_input = st.session_state.keyword_input_uncertain
    if keyword_input:
        keywords = keyword_input.strip().split()
    else:
        keywords = ["beginner"]
    
    st.session_state.keywords = keywords
    st.session_state.recommendations = recommend_courses(keywords)
    st.session_state.show_recommendations = True

def process_recommend_course_keywords():
    keyword_input = st.session_state.keyword_input_recommend
    if keyword_input:
        keywords = keyword_input.strip().split()
        st.session_state.keywords = keywords
        st.session_state.recommendations = recommend_courses(keywords)
        st.session_state.show_recommendations = True

# User input box
if user_input := st.chat_input("Please enter your question (e.g., Can you recommend me a course?)"):
    st.chat_message("You").write(user_input)

    # Top-3 intent recognition
    top_intents = predict_top_k_intents(user_input)
    top1_intent, top1_confidence = top_intents[0]
    st.session_state.chat_history.append(("You", user_input))
    
    # Reset recommendation display flag 
    st.session_state.show_recommendations = False
    st.session_state.intent_processed = False
    st.session_state.current_intent = top1_intent

    # Dialogue guidance and greeting logic
    if top1_intent == "greeting" and top1_confidence > 0.6:
        st.chat_message("🤖").write("Hello! Nice to meet you! What would you like to talk about?")
        st.session_state.intent_processed = True
    elif top1_intent == "goodbye" and top1_confidence > 0.6:
        st.chat_message("🤖").write("Alright, see you next time!")
        st.session_state.intent_processed = True
    elif top1_intent == "name" and top1_confidence > 0.6:
        st.chat_message("🤖").write("I am a chatbot. You can call me 'SmartBot'! ")
        st.session_state.intent_processed = True
    elif top1_intent == "uncertain_feeling" and top1_confidence > 0.5:
        st.chat_message("🤖").write("It sounds like you're a bit confused. How about I recommend some beginner courses for you? You can start with fields like AI or marketing! You can also enter your own interests below.")
        st.session_state.intent_processed = True
    elif top1_intent == "recommend_course" and top1_confidence > 0.7:
        st.chat_message("🤖").write(f"Great, I understand what you mean!")
        st.session_state.intent_processed = True
    elif top1_confidence < 0.6:
        st.chat_message("🤖").write("I'm not quite sure about your intent yet, but I guess you might mean one of these:")
        for intent, score in top_intents:
            st.markdown(f"- `{intent}` (Confidence {score:.2f})")
        st.markdown("Could you rephrase your question? I'll try my best to understand!")
        st.session_state.intent_processed = True
    else:
        st.chat_message("🤖").write(f"The intent I recognized is `{top1_intent}` (Confidence {top1_confidence:.2f}), but there is no executable action for now. Feel free to ask more questions!")
        st.session_state.intent_processed = True

# Show forms based on current intent if not processed yet
if st.session_state.current_intent == "uncertain_feeling" and not st.session_state.show_recommendations:
    with st.form(key="keyword_form_uncertain"):
        st.text_input(
            "What fields are you interested in? Please enter keywords (e.g., AI, finance, Python):",
            key="keyword_input_uncertain"
        )
        submit_button = st.form_submit_button("Get Recommendations", on_click=process_uncertain_feeling_keywords)

elif st.session_state.current_intent == "recommend_course" and not st.session_state.show_recommendations:
    with st.form(key="keyword_form_recommend"):
        st.text_input(
            "What fields are you interested in? Please enter keywords (e.g., AI, finance, Python):",
            key="keyword_input_recommend"
        )
        submit_button = st.form_submit_button("Get Recommendations", on_click=process_recommend_course_keywords)

# Display recommendations if available
if hasattr(st.session_state, 'show_recommendations') and st.session_state.show_recommendations and st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    if recommendations.empty:
        st.chat_message("🤖").write("No courses found for your keywords, please try different keywords.")
    else:
        st.chat_message("🤖").write("Here are some carefully selected courses for you:")
        st.dataframe(show_recommendation_table(recommendations))
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['Course Title']}**: {generate_reason(row)}")

# Show chat history
if st.checkbox("Show all chat history"):
    st.markdown("---")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")