import feedparser

# Use a pipeline as a high-level helper
from transformers import pipeline

ticker = 'AVGO'
keyword = 'BROADCOM'

pipe = pipeline("text-classification", model="ProsusAI/finbert")

rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

feed = feedparser.parse(rss_url)

total_score = 0
num_articles = 0

print(f'\nAnalyzing sentiment for {ticker} ({keyword})')
for i, entry in enumerate(feed.entries):
    if keyword.lower() not in entry.summary.lower():
        continue
    print('-' * 80)
    print(f'Title: {entry.title}')
    print(f'Summary: {entry.summary}')
    print(f'Link: {entry.link}')
    print(f'Published: {entry.published}')

    sentiment = pipe(entry.summary)[0]
    print(f'Sentiment: {sentiment["label"]} (Score: {sentiment["score"]:.4f})')

    if sentiment['label'] == 'positive':
        total_score += sentiment['score']
        num_articles += 1
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
        num_articles += 1

final_score = total_score / num_articles if num_articles > 0 else 0
print()
print('=' * 80)
print(f'FINAL SCORE - Sentiment for {ticker}: {"Positive" if final_score > 0.15 else "Negative" if final_score < 0.15 else "Neutral"} {final_score:.4f} after {num_articles} articles')
print('=' * 80,'\n')
