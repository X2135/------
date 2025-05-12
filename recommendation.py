import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 1. 加载课程数据
course_info = pd.read_csv("CourseraDataset-Clean.csv")

# ✅ 2. 查看字段结构（调试用）
print(course_info.columns)
print(course_info.head())

# ✅ 3. 只保留我们需要的字段
# ✅ 3. 只保留我们需要的字段（新增字段一起选）
standard = [
    'Course Title',
    'Modules',
    'Level',
    'Rating',
    'Keyword',
    'What you will learn',
    'Skill gain',
    'Duration to complete (Approx.)',
    'Number of Review'
]
true_data = course_info[standard].copy()
# ✅ 5. 删除重复值
true_data.drop_duplicates(subset='Course Title',inplace=True)
# ✅ 4. 删除缺失值
true_data.dropna(inplace=True)
# ✅ 6. 重置索引
true_data.reset_index(drop=True, inplace=True)



# ✅ 7. 英文语言检测逻辑（插入这里）
def english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# 用课程标题 + 模块组合进行检测
true_data = true_data[true_data['Course Title'].apply(english)]

# ✅ 8. 查看筛选后数据（调试用）
print("筛选后英文课程数量：", len(true_data))
print(true_data.sample(3))

true_data['semantic_text'] = (
    true_data['Course Title'].fillna('') + ' ' +
    true_data['Modules'].fillna('') + ' ' +
    true_data['What you will learn'].fillna('')
)

# ✅ 9. 初始化 BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ 10. 将 semantic_text 转为列表
transform_data = true_data['semantic_text'].tolist()

# ✅ 11. 编码所有课程为语义向量
course_embeddings = model.encode(transform_data, show_progress_bar=True)


def recommend_courses(user_input, preferred_level=None, top_k=5):
    if not user_input:
        return pd.DataFrame(columns=["Course Title", "Rating", "Level", "Keyword", "score", "similarity"])
    # 1. 拼接兴趣文本并向量化
    input_text = " ".join(user_input).lower()
    user_vec = model.encode([input_text])
    true_data['similarity'] = cosine_similarity(user_vec, course_embeddings)[0]

    # 2. 关键词匹配函数：计算重叠关键词数量
    def keyword_match_score(course_keyword):
        if pd.isna(course_keyword):
            return 0
        matches = sum(1 for word in user_input if word.lower() in course_keyword.lower())
        return min(0.1 * matches, 0.2)  # 最多加0.2分

    # 3. 综合打分函数
    def score(row):
        score = row['similarity']
        if preferred_level and preferred_level.lower() in row['Level'].lower():
            score += 0.1
        score += keyword_match_score(row['Keyword'])  # 关键词加权
        if row['Rating'] >= 4.5:
            score += 0.1
        if row['Number of Review'] >= 1000:
            score += 0.05
        return score

    # 4. 应用打分逻辑
    true_data['score'] = true_data.apply(score, axis=1)

    # 5. 排序推荐
    recommendations = true_data.sort_values(by='score', ascending=False).head(top_k)
    return recommendations

def generate_reason(row):
    reasons = []
    if row["similarity"] > 0.75:
        reasons.append("与你的兴趣高度匹配")
    if row["Rating"] >= 4.5:
        reasons.append("评分较高")
    if row["Number of Review"] > 1000:
        reasons.append("评价人数多")
    if "Advanced" in row["Level"]:
        reasons.append("适合进阶学习者")
    return "，".join(reasons) if reasons else "内容相关"

def show_recommendation_table(recommendations):
    result_table = pd.DataFrame(columns=[
        "课程名称", "推荐得分", "推荐理由", "匹配关键词", "语义匹配度",
        "评分", "评论数", "难度", "课程时长",
    ])

    for _, row in recommendations.iterrows():
        result_table = result_table.append({
            "课程名称": row['Course Title'],
            "推荐得分": round(row['score'], 3),
            "推荐理由": generate_reason(row),
            "匹配关键词": row['Keyword'],
            "语义匹配度": round(row['similarity'], 3),
            "评分": row['Rating'],
            "评论数": int(row['Number of Review']),
            "难度": row['Level'],
            "课程时长": row['Duration to complete (Approx.)'],
        }, ignore_index=True)

    return result_table

