# Browser History Analyzer

A comprehensive browser history analysis tool with an interactive Streamlit dashboard.

## Features

- Extract browser history from Chrome/Safari
- Categorize websites into different categories (productivity, social media, entertainment, etc.)
- Domain clustering and pattern analysis
- **Accurate time tracking** with deduplication and smart capping
- Interactive dashboard with filters and visualizations
- Week-over-week insights and comparisons

### Time Tracking Accuracy

The analyzer uses smart algorithms to provide realistic browsing time estimates:
- **Duration Capping**: Individual visits are capped at 2 hours to exclude tabs left open overnight
- **Overlap Deduplication**: When multiple tabs are open simultaneously, the time is only counted once
- This prevents impossible metrics like "104 hours/day" and gives you realistic daily averages

## Installation

1) Install Python 3.8+ (already included on most systems).
2) Get the code:
```bash
git clone https://github.com/ali-lari1/browser-history-analyzer.git
cd browser-history-analyzer
```
3) Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract Browser History

Point the extractor to your browser’s history DB (or copy it into this folder and rename it).

- **Chrome (macOS)**: `~/Library/Application Support/Google/Chrome/Default/History`
- **Chrome (Windows)**: `%LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default\\History`
- **Chrome (Linux)**: `~/.config/google-chrome/Default/History`
- **Brave (macOS)**: `~/Library/Application Support/BraveSoftware/Brave-Browser/Default/History`
- **Brave (Windows)**: `%LOCALAPPDATA%\\BraveSoftware\\Brave-Browser\\User Data\\Default\\History`
- **Edge (Windows)**: `%LOCALAPPDATA%\\Microsoft\\Edge\\User Data\\Default\\History`

Two options:
1) **Copy the file** into the repo and name it `browser_history.db` (default expected by `extraction_script.py`), then run:

```bash
python extraction_script.py
```

2) **Or update the path** in `extraction_script.py`:
```python
conn = sqlite3.connect("/absolute/path/to/your/History")
```
Close your browser first or copy the file elsewhere; the live DB is locked while the browser is running.

This creates `full_history.csv` with your browsing data.

### 2. Cluster Domains

```bash
python cluster_domains.py
```

This groups similar domains together and creates `clustered_history.csv`.

### 3. Analyze Patterns

```bash
python analyze_patterns.py
```

This categorizes websites and creates `categorized_history.csv`.

### 4. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Dashboard Features

### Interactive Filters
- **Date Range**: Select specific date ranges to analyze
- **Categories**: Filter by website categories (productivity, social media, entertainment, etc.)
- **Domains**: Focus on specific domains from the top 20

### Visualizations
- **Browsing Time Over Time**: Line chart showing daily browsing patterns
- **Time by Category**: Bar chart of total time spent per category
- **Hourly Activity Pattern**: See which hours you're most active
- **Top 10 Domains**: Pie chart of your most visited domains
- **Weekday vs Weekend**: Compare browsing habits
- **Activity Heatmap**: Day of week vs hour visualization

### Insights
- Total browsing time and statistics
- Average daily usage
- Top categories and domains
- Week-over-week comparisons showing percentage changes
- Category-specific trend analysis (e.g., "You spent 40% more time on YouTube this week")

## Data Files

- `full_history.csv`: Raw extracted browser history
- `clustered_history.csv`: History with domain clustering
- `categorized_history.csv`: History with categories (used by dashboard)

## Requirements

- Python 3.8+
- pandas
- streamlit
- plotly
- numpy
