import joblib
import streamlit as st
from recommendation import recommend_courses

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

# Context memory (store keywords)
if "context" not in st.session_state:
    st.session_state.context = {}

# User input box
if user_input := st.chat_input("Please enter your question (e.g., Can you recommend me a course?)"):
    st.chat_message("You").write(user_input)

    # Top-3 intent recognition
    top_intents = predict_top_k_intents(user_input)
    top1_intent, top1_confidence = top_intents[0]
    st.session_state.chat_history.append(("You", user_input))

    # Dialogue guidance and greeting logic
    if top1_intent == "greeting" and top1_confidence > 0.6:
        st.chat_message("🤖").write("Hello! Nice to meet you! I can recommend courses, introduce features, or answer questions about the platform. What would you like to talk about?")
    elif top1_intent == "goodbye" and top1_confidence > 0.6:
        st.chat_message("🤖").write("Alright, see you next time! Happy learning! 😊")
    elif top1_intent == "name" and top1_confidence > 0.6:
        st.chat_message("🤖").write("I am a chatbot focused on course recommendations. You can call me 'SmartBot'! Feel free to ask me for course advice anytime.")
    elif top1_intent == "uncertain_feeling" and top1_confidence > 0.5:
        st.chat_message("🤖").write("It sounds like you're a bit confused. How about I recommend some beginner courses for you? You can start with fields like AI or marketing!")
        st.session_state.context["keywords"] = ["beginner"]

    # Course recommendation intent handling
    elif top1_intent == "recommend_course" and top1_confidence > 0.7:
        st.chat_message("🤖").write(f"Great, I understand what you mean!")

        # Use existing keywords or prompt for input
        if "keywords" in st.session_state.context:
            keywords = st.session_state.context["keywords"]
            st.chat_message("🤖").write(f"The keywords you mentioned earlier are: `{' '.join(keywords)}`. Let me check for you!")
        else:
            keyword_input = st.text_input("What fields are you interested in? Please enter keywords (e.g., AI, finance, Python):", key="keyword_input")
            if keyword_input:
                keywords = keyword_input.strip().split()
                st.session_state.context["keywords"] = keywords

        if "keywords" in st.session_state.context:
            recommendations = recommend_courses(st.session_state.context["keywords"])
            st.chat_message("🤖").write("Here are some carefully selected courses for you:")
            for _, row in recommendations.iterrows():
                with st.expander(f"📘 {row['Course Title']} (Rating {row['Rating']})"):
                    st.markdown(f"""
                    **Keyword Match:** {row['Keyword']}  
                    **Similarity:** {round(row['similarity'], 2)}  
                    **Level:** {row['Level']}  
                    **Duration:** {row['Duration to complete (Approx.)']}  
                    **Number of Reviews:** {int(row['Number of Review'])}  
                    """)
    elif top1_confidence < 0.6:
        st.chat_message("🤖").write("I'm not quite sure about your intent yet, but I guess you might mean one of these:")
        for intent, score in top_intents:
            st.markdown(f"- `{intent}` (Confidence {score:.2f})")
        st.markdown("Could you rephrase your question? I'll try my best to understand!")
    else:
        st.chat_message("🤖").write(f"The intent I recognized is `{top1_intent}` (Confidence {top1_confidence:.2f}), but there is no executable action for now. Feel free to ask more questions!")

# Show chat history
if st.checkbox("Show all chat history"):
    st.markdown("---")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")