import joblib
import streamlit as st
from recommendation import recommend_courses

# ✅ 加载模型和向量器
classifier = joblib.load("intent_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ✅ 意图识别函数（Top-K 排序）
def predict_top_k_intents(user_input, top_k=3):
    user_vec = vectorizer.transform([user_input])
    probs = classifier.predict_proba(user_vec)[0]
    top_indices = probs.argsort()[::-1][:top_k]
    top_intents = [(classifier.classes_[i], probs[i]) for i in top_indices]
    return top_intents

# ✅ 页面设置
st.set_page_config(page_title="🎓 智能课程助手", layout="centered")
st.title("🤖 智能课程推荐聊天机器人")

# ✅ 聊天记录状态管理
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ 上下文记忆（存储关键词）
if "context" not in st.session_state:
    st.session_state.context = {}

# ✅ 用户输入框
if user_input := st.chat_input("请输入你的问题（如：Can you recommend me a course?）"):
    st.chat_message("你").write(user_input)

    # ✅ Top-3 意图识别
    top_intents = predict_top_k_intents(user_input)
    top1_intent, top1_confidence = top_intents[0]
    st.session_state.chat_history.append(("你", user_input))

    # ✅ 对话前引导和寒暄逻辑
    if top1_intent == "greeting" and top1_confidence > 0.6:
        st.chat_message("🤖").write("你好呀～很高兴见到你！我可以帮你推荐课程、介绍功能，或回答关于平台的问题哦。你想聊点什么？")
    elif top1_intent == "goodbye" and top1_confidence > 0.6:
        st.chat_message("🤖").write("好哒，那我们下次再见啦，祝你学习愉快！😊")
    elif top1_intent == "name" and top1_confidence > 0.6:
        st.chat_message("🤖").write("我是一位专注于课程推荐的聊天机器人，大家都叫我“小智”！你可以随时问我选课的建议～")
    elif top1_intent == "uncertain_feeling" and top1_confidence > 0.5:
        st.chat_message("🤖").write("听起来你有点迷茫呢～不如我帮你推荐一些入门课程？可以从 AI、marketing 等领域开始试试噢！")
        st.session_state.context["keywords"] = ["beginner"]

    # ✅ 课程推荐意图处理
    elif top1_intent == "recommend_course" and top1_confidence > 0.7:
        st.chat_message("🤖").write(f"太好了，我懂你的意思了！")

        # 使用已有关键词或提示输入
        if "keywords" in st.session_state.context:
            keywords = st.session_state.context["keywords"]
            st.chat_message("🤖").write(f"你之前提到的关键词是：`{' '.join(keywords)}`，我这就来查一查～")
        else:
            keyword_input = st.text_input("你感兴趣的领域有哪些？请填写关键词（如 AI、finance、Python）：", key="keyword_input")
            if keyword_input:
                keywords = keyword_input.strip().split()
                st.session_state.context["keywords"] = keywords

        if "keywords" in st.session_state.context:
            recommendations = recommend_courses(st.session_state.context["keywords"])
            st.chat_message("🤖").write("以下是我为你精挑细选的课程：")
            for _, row in recommendations.iterrows():
                with st.expander(f"📘 {row['Course Title']}（评分 {row['Rating']}）"):
                    st.markdown(f"""
                    **关键词匹配：** {row['Keyword']}  
                    **匹配度：** {round(row['similarity'], 2)}  
                    **难度：** {row['Level']}  
                    **学习时长：** {row['Duration to complete (Approx.)']}  
                    **评论数：** {int(row['Number of Review'])}  
                    """)
    elif top1_confidence < 0.6:
        st.chat_message("🤖").write("我还不太确定你的意图呢～不过我猜测你可能是想表达这些意思：")
        for intent, score in top_intents:
            st.markdown(f"- `{intent}`（置信度 {score:.2f}）")
        st.markdown("你可以稍微换个说法吗？我会努力理解的！")
    else:
        st.chat_message("🤖").write(f"我识别到的意图是 `{top1_intent}`（置信度 {top1_confidence:.2f}），暂时没有可执行的操作～欢迎你继续提问哦！")


# ✅ 显示聊天记录
if st.checkbox("展开全部聊天记录"):
    st.markdown("---")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}：** {msg}")