import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from urllib.parse import urlparse

# Connect to Brave history
conn = sqlite3.connect('brave_history.db')

# Extract ALL history with timestamps
query = """
SELECT
    urls.url,
    urls.title,
    visits.visit_time,
    visits.visit_duration
FROM urls
JOIN visits ON urls.id = visits.url
ORDER BY visits.visit_time DESC
"""

df = pd.read_sql_query(query, conn)
conn.close()

# Conver Chrome timestamps (microseconds since 1601-01-01) to human-readable format
def chrome_time_to_datetime(chrome_time):
    epoch_start = datetime(1601, 1, 1)
    return epoch_start + timedelta(microseconds=chrome_time)

df['datetime'] = df['visit_time'].apply(chrome_time_to_datetime)
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['day of week'] = df['datetime'].dt.day_name()

# Extract domain from URL
df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)

# Save to CSV
df.to_csv('full_history.csv', index=False)

print(f"\nExtracted {len(df)} browsin g records!")
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")


print("\nTop 10 visited domains:")
print(df['domain'].value_counts().head(10))

print("\nBusiest hours:")
print(df['hour'].value_counts().sort_index().head(10))

print("\nDaily browsing:")
daily = df.groupby('date').size()
print(f"Average: {daily.mean():.0f} pages/day")
print(f"Peak day: {daily.max()} pages on {daily.idxmax()}")