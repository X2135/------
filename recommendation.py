import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity


df_courses = pd.read_csv("CourseraDataset-Clean.csv")


print(df_courses.columns)
print(df_courses.head())

# 3. Keep needed columns
columns_needed = [
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
course_data = df_courses[columns_needed].copy()
# Remove duplicates
course_data.drop_duplicates(subset='Course Title', inplace=True)
# Remove missing values
course_data.dropna(inplace=True)
# Reset index
course_data.reset_index(drop=True, inplace=True)


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# only English courses
df_english = course_data[course_data['Course Title'].apply(is_english)]


print("Number of English courses after filtering:", len(df_english))
print(df_english.sample(3))

df_english['semantic_text'] = (
    df_english['Course Title'].fillna('') + ' ' +
    df_english['Modules'].fillna('') + ' ' +
    df_english['What you will learn'].fillna('')
)

#BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert semantic_text to list
semantic_texts = df_english['semantic_text'].tolist()

# Encode all courses as semantic vectors
course_embeddings = model.encode(semantic_texts, show_progress_bar=True)

def recommend_courses(user_input, preferred_level=None, top_k=5):
    if not user_input:
        return pd.DataFrame(columns=["Course Title", "Rating", "Level", "Keyword", "score", "similarity"])
    #Join interest text and vectorize
    input_text = " ".join(user_input).lower()
    user_vec = model.encode([input_text])
    df_english['similarity'] = cosine_similarity(user_vec, course_embeddings)[0]

    # Keyword match score (partial match)
    def keyword_match_score(course_keyword):
        if pd.isna(course_keyword):
            return 0
        matches = sum(1 for word in user_input if word.lower() in course_keyword.lower())
        return min(0.1 * matches, 0.2)  # max 0.2

    # 3. Scoring function
    def score(row):
        score = row['similarity']
        if preferred_level and preferred_level.lower() in row['Level'].lower():
            score += 0.1
        score += keyword_match_score(row['Keyword'])
        if row['Rating'] >= 4.5:
            score += 0.1
        if row['Number of Review'] >= 1000:
            score += 0.05
        return score

    df_english['score'] = df_english.apply(score, axis=1)

    recommendations = df_english.sort_values(by='score', ascending=False).head(top_k)
    return recommendations

def generate_reason(row):
    reasons = []
    if row["similarity"] > 0.75:
        reasons.append("High semantic match with your interests")
    if row["Rating"] >= 4.5:
        reasons.append("High rating")
    if row["Number of Review"] > 1000:
        reasons.append("Many reviews")
    if "Advanced" in row["Level"]:
        reasons.append("Suitable for advanced learners")
    return ", ".join(reasons) if reasons else "Relevant content"

def show_recommendation_table(recommendations):
    result_table = pd.DataFrame(columns=[
        "Course Title", "Score", "Reason", "Keyword Match", "Semantic Similarity",
        "Rating", "Number of Reviews", "Level", "Duration",
    ])

    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        result_table = result_table.append({
            "Course Title": row['Course Title'],
            "Score": round(row['score'], 3),
            "Reason": generate_reason(row),
            "Keyword Match": row['Keyword'],
            "Semantic Similarity": round(row['similarity'], 3),
            "Rating": row['Rating'],
            "Number of Reviews": int(row['Number of Review']),
            "Level": row['Level'],
            "Duration": row['Duration to complete (Approx.)'],
        }, ignore_index=True)

    # Reset index to start from 1 instead of 0
    result_table.index = result_table.index + 1
    
    return result_table

