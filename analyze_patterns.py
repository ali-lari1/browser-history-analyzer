import pandas as pd
from collections import defaultdict
import re

# Load your extracted history
df = pd.read_csv('full_history.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Check if clustering data exists, if not run clustering
import os
if not os.path.exists('clustered_history.csv'):
    print("ℹ Clustering data not found. Running domain clustering first...")
    import subprocess
    subprocess.run(['python', 'cluster_domains.py'], check=True)
    print()

# Load clustered data
clustered_df = pd.read_csv('clustered_history.csv')
df['cluster'] = clustered_df['cluster']
df['normalized_domain'] = clustered_df['normalized_domain']

# Category mapping
CATEGORIES = {
    'productivity': ['github', 'stackoverflow', 'docs.', 'notion', 'linear', 'jira', 'trello', 
                     'asana', 'monday', 'confluence', 'figma', 'miro', 'coda'],
    'learning': ['coursera', 'udemy', 'edx', 'khanacademy', 'youtube.com/watch', 'medium', 
                 'towardsdatascience', 'arxiv', 'scholar.google', 'wikipedia'],
    'social': ['twitter', 'x.com', 'facebook', 'instagram', 'linkedin', 'reddit', 
               'discord', 'slack', 'whatsapp', 'telegram'],
    'entertainment': ['youtube.com', 'netflix', 'spotify', 'twitch', 'tiktok', 
                      'hulu', 'disneyplus', 'primevideo'],
    'news': ['nytimes', 'wsj', 'bbc', 'cnn', 'reuters', 'bloomberg', 'theguardian',
             'techcrunch', 'hackernews', 'news.ycombinator'],
    'shopping': ['amazon', 'ebay', 'etsy', 'walmart', 'bestbuy', 'target', 'aliexpress'],
    'ai_ml': ['openai', 'anthropic', 'claude.ai', 'chatgpt', 'huggingface', 'kaggle',
              'paperswithcode', 'deepmind', 'stability'],
    'finance': ['yahoo.finance', 'tradingview', 'investing.com', 'marketwatch', 
                'coinbase', 'binance', 'robinhood']
}

def categorize_domain(domain):

    domain_lower = domain.lower() 

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in domain_lower:
                return category
    
    return 'other'

# Apply categorization 
df['category'] = df['domain'].apply(categorize_domain)

# Time-based analysis
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['datetime'].dt.isocalendar().week
df['month'] = df['datetime'].dt.month
df['is_weekend'] = df['datetime'].dt.dayofweek >= 5

print("\n" + "="*60)
print("BBROWSER BEHAVIOR ANALYSIS")
print("="*60)

# Category breakdown
print("\nTime spent by category:")
category_counts = df['category'].value_counts()
for cat, count in category_counts.items():
    pct = (count / len(df)) * 100
    print(f"{cat.upper():15} {count:6,} visits ({pct:5.1f}%)")

# Weekday vs weekend
print("\nWeekday vs weekend browsing:")
weekday_avg = df[~df['is_weekend']].groupby('date').size().mean()
weekend_avg = df[df['is_weekend']].groupby('date').size().mean()
print(f"Weekday average: {weekday_avg:.0f} pages/day")
print(f"Weekend average: {weekend_avg:.0f} pages/day")
print(f"Difference: {((weekend_avg - weekday_avg) / weekday_avg) * 100:+.0f}%")

# Peak browsing times
print("\n Peak browsing hours:")
hourly = df.groupby('hour').size().sort_values(ascending=False)
for hour, count in hourly.head(5).items(): 
    print(f"{hour:02d}:00 - {hour+1:02d}:00  →  {count:,} pages")

# Most productive vs least productive days
print("\nProductivity patterns:")
daily_productivity = df[df['category'].isin(['productivity', 'learning', 'finance', 'ai_ml'])].groupby('date').size()
if len(daily_productivity) > 0:
    print(f"Best day: {daily_productivity.max()} productive visits on {daily_productivity.idxmax()}")
    print(f"Average: {daily_productivity.mean():.1f} productive visits/day")

# Doomscroll vs productivity ratio
productivity_count = len(df[df['category'].isin(['productivity', 'learning', 'finance', 'ai_ml'])])
entertainment_count = len(df[df['category'].isin(['entertainment', 'social', 'news', 'shopping'])])

if productivity_count > 0:
    ratio = entertainment_count / productivity_count
    print(f"\nDoomscroll vs. productivity ratio: {ratio:.2f}")
    if ratio > 2:
        print("You doomscroll more than twice as much as you are productive. Consider balancing your browsing habits!")
    elif ratio < 0.75:
        print("Great job! Your productive browsing outweighs doomscrolling.")

# Recent trends (last 30 days vs previous 30 days)
last_30 = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
previous_30 = df[(df['date'] < df['date'].max() - pd.Timedelta(days=30)) & 
                 (df['date'] >= df['date'].max() - pd.Timedelta(days=60))]

if len(previous_30) > 0 and len(last_30) > 0:
    print("\nLast 30 days vs previous 30 days:")
    recent_daily = len(last_30) / 30
    previous_daily = len(previous_30) / 30
    change_pct = ((recent_daily - previous_daily) / previous_daily) * 100
    print(f"Recent: {recent_daily:.0f} pages/day")
    print(f"Previous {previous_daily:.0f} pages/day")
    print(f"Change: {change_pct:+.0f}%")

# Save categorized data
df.to_csv('categorized_history.csv', index=False)
print("\nCategorized browsing data saved to 'categorized_history.csv'")

# Generate insights summary
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

top_category = category_counts.index[0]
print(f"1. You spend the most time on: {top_category.upper()}")

if df['is_weekend'].sum() > 0:
    weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
    if weekend_ratio > 1.2:
        print(f"2. You browse {((weekend_ratio - 1) * 100):.0f}% MORE on weekends")
    elif weekend_ratio < 0.8:
        print(f"2. You browse {((1 - weekend_ratio) * 100):.0f}% LESS on weekends")

peak_hour = hourly.index[0]
print(f"3. Peak browsing: {peak_hour:02d}:00-{peak_hour+1:02d}:00")

# Cluster-based insights
print("\n" + "="*60)
print("BROWSING CLUSTERS (ML-DISCOVERED PATTERNS)")
print("="*60)

# Get cluster visit counts (excluding unclustered -1)
cluster_data = df[df['cluster'] != -1].groupby('cluster').agg({
    'url': 'count',
    'domain': lambda x: x.nunique(),
    'hour': lambda x: x.mode()[0] if len(x) > 0 else 0
}).rename(columns={'url': 'visits', 'domain': 'unique_domains', 'hour': 'peak_hour'})

cluster_data = cluster_data.sort_values('visits', ascending=False)

for cluster_id, row in cluster_data.iterrows():
    # Get top domains in this cluster
    cluster_domains = df[df['cluster'] == cluster_id]['domain'].value_counts().head(3)
    top_domains_str = ', '.join([f"{domain} ({count})" for domain, count in cluster_domains.items()])

    print(f"\nCluster {cluster_id + 1}: {int(row['visits'])} visits across {int(row['unique_domains'])} domains")
    print(f"  Peak hour: {int(row['peak_hour']):02d}:00")
    print(f"  Top sites: {top_domains_str}")

# Most productive cluster
productive_clusters = df[df['category'].isin(['productivity', 'learning', 'finance', 'ai_ml']) & (df['cluster'] != -1)]
if len(productive_clusters) > 0:
    most_productive_cluster = productive_clusters['cluster'].mode()[0]
    productive_count = len(productive_clusters[productive_clusters['cluster'] == most_productive_cluster])
    print(f"\n→ Most productive cluster: Cluster {most_productive_cluster + 1} ({productive_count} productive visits)")

# Entertainment cluster
entertainment_clusters = df[df['category'].isin(['entertainment', 'social', 'shopping', 'news']) & (df['cluster'] != -1)]
if len(entertainment_clusters) > 0:
    most_entertainment_cluster = entertainment_clusters['cluster'].mode()[0]
    entertainment_count = len(entertainment_clusters[entertainment_clusters['cluster'] == most_entertainment_cluster])
    print(f"→ Most entertainment cluster: Cluster {most_entertainment_cluster + 1} ({entertainment_count} entertainment visits)")

print("\n")