plt.figure(figsize=(10, 5))
sns.countplot(x='label', data=df,
            order=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])
plt.title("Sentiment")
plt.ylabel("Count", fontsize=12)
plt.xlabel("Sentiments", fontsize=12)
plt.show()