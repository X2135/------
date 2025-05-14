import matplotlib.pyplot as plt
import pandas as pd
from recommendation import recommend_courses, show_recommendation_table

# Define test keyword sets
TEST_KEYWORDS = [
    ["AI", "machine learning"],
    ["finance", "investment"],
    ["marketing"],
    ["data science"]
]

# Adjusted column widths for compactness and no overflow
col_widths = {
    "Recommendation For": 0.16,
    "Course Title": 0.35,
    "Score": 0.08,
    "Keyword Match": 0.14,
    "Semantic Similarity": 0.16,
    "Rating": 0.09
}

# Collect all results into a single DataFrame
all_results = []
for keywords in TEST_KEYWORDS:
    recommendations = recommend_courses(keywords)
    table = show_recommendation_table(recommendations)
    table.insert(0, "Recommendation For", ', '.join(keywords))
    # Remove unwanted columns
    table = table.drop(columns=["Reason", "Duration", "Number of Reviews", "Level"])
    all_results.append(table)

big_table = pd.concat(all_results, ignore_index=True)

fig, ax = plt.subplots(figsize=(11, min(0.38 * len(big_table) + 2, 18)))
ax.axis('off')

# Beautify: alternating row colors, bold headers, grid lines
row_colors = ["#f8f8f8" if i % 2 == 0 else "#eaeaea" for i in range(len(big_table))]
cell_colours = [[row_colors[i]] * len(big_table.columns) for i in range(len(big_table))]
tbl = ax.table(
    cellText=big_table.values,
    colLabels=big_table.columns,
    cellLoc='center',
    loc='center',
    cellColours=cell_colours
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.0)

# Bold header
for key, cell in tbl.get_celld().items():
    row, col = key
    if row == 0:
        cell.set_text_props(weight='bold', color='#222')
        cell.set_facecolor('#d0d0d0')
    cell.set_linewidth(0.7)
    cell.set_edgecolor('#888')

# Set custom column widths
for i, col in enumerate(big_table.columns):
    width = col_widths.get(col, 0.1)
    for row in range(len(big_table) + 1):  # +1 for header
        tbl[(row, i)].set_width(width)

plt.title("All Recommendations Table", fontsize=16, weight='bold', pad=10)
plt.tight_layout(pad=0.3)
plt.savefig("all_recommendations_table.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: all_recommendations_table.png") 