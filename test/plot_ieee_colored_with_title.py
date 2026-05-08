import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set IEEE compliant style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Data
total = 527
labels = ['Positive', 'Neutral', 'Negative']

finbert_pct = [75/total*100, 416/total*100, 36/total*100]
hybrid_pct = [268/total*100, 182/total*100, 77/total*100]

x = np.arange(len(labels))
width = 0.35  # width of the bars

# Make figure slightly taller to accommodate the title well
fig, ax = plt.subplots(figsize=(5.5, 4.0))

# Professional colored format (Blue and Orange)
color_base = '#4C72B0'  # Soft Blue
color_prop = '#DD8452'  # Soft Orange

# Plot bars without hatch
rects1 = ax.bar(x - width/2, finbert_pct, width, label='Original FinBERT', color=color_base, edgecolor='black')
rects2 = ax.bar(x + width/2, hybrid_pct, width, label='Hybrid FinBERT', color=color_prop, edgecolor='black')

# Labeling
ax.set_ylabel('Percentage of News Articles (%)')

# Add Chart Title
ax.set_title('Sentiment Distribution of News Articles:\nOriginal FinBERT vs. Hybrid FinBERT')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 105) # Add a bit of space at top for labels
ax.legend(loc='upper right')

# Add percentage text on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontname='Times New Roman')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Save the plot
output_file = 'sentiment_distribution_colored_title.png'
plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
print(f"Chart saved as {output_file}")
