import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix


# Load your browsing history
df = pd.read_csv('full_history.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Normalize domains: remove 'www.' prefix and trailing dots
def normalize_domain(domain):
    if pd.isna(domain):
        return domain
    domain = str(domain).lower().strip()
    # Remove www. prefix
    if domain.startswith('www.'):
        domain = domain[4:]
    # Remove trailing dots
    domain = domain.rstrip('.')
    return domain

df['normalized_domain'] = df['domain'].apply(normalize_domain)

print("\n" + "="*60)
print("DOMAIN CLUSTERING ANALYSIS")
print("="*60)

# Get domain visit counts (using normalized domains)
domain_stats = df.groupby('normalized_domain').agg({
    'url': 'count',  # visit count
    'hour': lambda x: x.mode()[0] if len(x) > 0 else 0,  # most common hour
    'datetime': lambda x: (x.max() - x.min()).days  # span of days visited
}).rename(columns={'url': 'visit_count', 'hour': 'peak_hour', 'datetime': 'day_span'})

# Filter to domains with at least 3 visits
significant_domains = domain_stats[domain_stats['visit_count'] >= 3].copy()

if len(significant_domains) < 5:
    print("Not enough data for clustering. Need at least 5 domains with 3+ visits.")
    exit()

print(f"\nAnalyzing {len(significant_domains)} domains with 3+ visits...")

# Create features for clustering
# 1. Text-based features from domain names
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=50)
domain_text_features = vectorizer.fit_transform(significant_domains.index)

# 2. Behavioral features
behavioral_features = significant_domains[['visit_count', 'peak_hour', 'day_span']].values

# Normalize behavioral features to 0-1 scale
scaler = MinMaxScaler()
behavioral_features_scaled = scaler.fit_transform(behavioral_features)

# Weight adjustment: Give 3x more importance to behavior than text
# Multiply behavioral features by 3 to increase their influence
behavioral_features_weighted = behavioral_features_scaled * 3

# Combine text and behavioral features (behavior now has 3x weight)
combined_features = hstack([domain_text_features, csr_matrix(behavioral_features_weighted)])

# Determine optimal number of clusters (between 3 and 8)
n_clusters = min(max(3, len(significant_domains) // 5), 8)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
significant_domains['cluster'] = kmeans.fit_predict(combined_features)

# Analyze each cluster
print(f"\nDiscovered {n_clusters} browsing patterns:\n")

for cluster_id in range(n_clusters):
    cluster_domains = significant_domains[significant_domains['cluster'] == cluster_id]

    # Get stats for this cluster
    total_visits = cluster_domains['visit_count'].sum()
    avg_visits = cluster_domains['visit_count'].mean()
    common_hour = cluster_domains['peak_hour'].mode()[0] if len(cluster_domains) > 0 else "N/A"

    print(f"CLUSTER {cluster_id + 1}: {len(cluster_domains)} domains, {total_visits:,} total visits")
    print(f"  Average visits per domain: {avg_visits:.1f}")
    print(f"  Most active hour: {common_hour:02d}:00-{common_hour+1:02d}:00")

    # Show top domains in this cluster
    top_domains = cluster_domains.nlargest(5, 'visit_count')
    print("  Top domains:")
    for domain, row in top_domains.iterrows():
        print(f"    • {domain} ({row['visit_count']} visits)")
    print()

# Map clusters back to full dataframe (using normalized domains)
domain_to_cluster = significant_domains['cluster'].to_dict()
df['cluster'] = df['domain'].map(domain_to_cluster).fillna(-1).astype(int)

# Save clustered data
df.to_csv('clustered_history.csv', index=False)
print(f"Clustered browsing data saved to 'clustered_history.csv'")

# Find interesting patterns
print("\n" + "="*60)
print("CLUSTERING INSIGHTS")
print("="*60)

# Which cluster dominates which time of day?
for cluster_id in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 0:
        hourly_dist = cluster_data.groupby('hour').size()
        if len(hourly_dist) > 0:
            peak_hour = hourly_dist.idxmax()
            peak_count = hourly_dist.max()
            if peak_count > 20:  # Only show if significant
                print(f"Cluster {cluster_id + 1} peaks at {peak_hour:02d}:00 ({peak_count} visits)")

# Weekend vs weekday preferences by cluster
df['is_weekend'] = pd.to_datetime(df['datetime']).dt.dayofweek >= 5
for cluster_id in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 30:  # Only analyze clusters with enough data
        weekend_pct = (cluster_data['is_weekend'].sum() / len(cluster_data)) * 100
        if weekend_pct > 60:
            print(f"Cluster {cluster_id + 1} is {weekend_pct:.0f}% weekend browsing")
        elif weekend_pct < 20:
            print(f"Cluster {cluster_id + 1} is {100-weekend_pct:.0f}% weekday browsing")
        else:
            print(f"Cluster {cluster_id + 1} has balanced browsing ({weekend_pct:.0f}% weekend)")

print("\n")
