import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Browser History Dashboard",
    page_icon="🌐",
    layout="wide"
)

# Helper function to format time based on selected unit
def format_time(minutes, unit='hours'):
    """Format time in the specified unit"""
    if unit == 'minutes':
        return minutes, 'min'
    elif unit == 'hours':
        return minutes / 60, 'hrs'
    elif unit == 'days':
        return minutes / 60 / 24, 'days'
    return minutes / 60, 'hrs'  # default to hours

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('categorized_history.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = pd.to_datetime(df['date'])

    # Use deduplicated time if available, otherwise calculate duration from raw data
    if 'deduplicated_minutes' in df.columns:
        # Deduplicated time is per-day, so we use it for daily metrics
        # For individual visit displays, still show the raw duration
        df['duration_minutes'] = df['visit_duration'] / 1000000 / 60
    else:
        # Fallback to raw duration
        df['duration_minutes'] = df['visit_duration'] / 1000000 / 60
        # Add a warning that data should be regenerated
        st.sidebar.warning("⚠️ Data uses overlapping durations. Re-run analyze_patterns.py for accurate time tracking.")

    return df

df = load_data()

# Title
st.title("🌐 Browser History Analytics Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")

# Time unit selector
time_unit = st.sidebar.selectbox(
    "Time Display Unit",
    options=['minutes', 'hours', 'days'],
    index=1  # default to hours
)

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle single date selection
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

# Category filter
categories = ['All'] + sorted(df['category'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=categories,
    default=['All']
)

# Domain filter
top_domains = df['domain'].value_counts().head(20).index.tolist()
selected_domains = st.sidebar.multiselect(
    "Select Domains (top 20)",
    options=['All'] + top_domains,
    default=['All']
)

# Apply filters
filtered_df = df[
    (df['date'].dt.date >= start_date) &
    (df['date'].dt.date <= end_date)
]

if 'All' not in selected_categories:
    filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

if 'All' not in selected_domains:
    filtered_df = filtered_df[filtered_df['domain'].isin(selected_domains)]

# Calculate insights
st.sidebar.markdown("---")
st.sidebar.header("Quick Stats")
st.sidebar.metric("Total Visits", f"{len(filtered_df):,}")

# Use deduplicated time if available
if 'deduplicated_minutes' in filtered_df.columns:
    # Sum deduplicated minutes across unique days
    total_time_dedup = filtered_df.groupby('date')['deduplicated_minutes'].first().sum()
    total_time_val, total_time_unit = format_time(total_time_dedup, time_unit)
else:
    total_time_val, total_time_unit = format_time(filtered_df['duration_minutes'].sum(), time_unit)

st.sidebar.metric("Total Time", f"{total_time_val:.1f} {total_time_unit}")
st.sidebar.metric("Unique Domains", f"{filtered_df['domain'].nunique()}")

# Main dashboard
if len(filtered_df) == 0:
    st.warning("No data available for the selected filters.")
else:
    # Insights section
    st.header("📊 Key Insights")

    # Calculate weekly comparison insights
    current_week = filtered_df[filtered_df['week'] == filtered_df['week'].max()]
    previous_week = filtered_df[filtered_df['week'] == filtered_df['week'].max() - 1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Use deduplicated time if available
        if 'deduplicated_minutes' in filtered_df.columns:
            total_time = filtered_df.groupby('date')['deduplicated_minutes'].first().sum()
        else:
            total_time = filtered_df['duration_minutes'].sum()

        total_val, total_unit = format_time(total_time, time_unit)
        st.metric(
            "Total Browsing Time",
            f"{total_val:.1f} {total_unit}"
        )

    with col2:
        # Use deduplicated time for average daily
        if 'deduplicated_minutes' in filtered_df.columns:
            avg_daily = filtered_df.groupby('date')['deduplicated_minutes'].first().mean()
        else:
            avg_daily = filtered_df.groupby('date')['duration_minutes'].sum().mean()

        avg_val, avg_unit = format_time(avg_daily, time_unit)
        st.metric(
            "Avg Daily Time",
            f"{avg_val:.1f} {avg_unit}/day"
        )

    with col3:
        most_visited_cat = filtered_df['category'].value_counts().index[0]
        cat_count = filtered_df['category'].value_counts().iloc[0]
        st.metric(
            "Top Category",
            most_visited_cat.capitalize()
        )
        st.markdown(f"<span style='background-color: #262730; padding: 4px 12px; border-radius: 16px; font-size: 14px; color: #FAFAFA;'>{cat_count:,} visits</span>", unsafe_allow_html=True)

    with col4:
        most_visited_domain = filtered_df['domain'].value_counts().index[0]
        domain_count = filtered_df['domain'].value_counts().iloc[0]
        st.metric(
            "Top Domain",
            most_visited_domain
        )
        st.markdown(f"<span style='background-color: #262730; padding: 4px 12px; border-radius: 16px; font-size: 14px; color: #FAFAFA;'>{domain_count:,} visits</span>", unsafe_allow_html=True)

    # Weekly comparison insights
    if len(current_week) > 0 and len(previous_week) > 0:
        st.markdown("### 📈 Weekly Trends")

        insight_cols = st.columns(2)

        # Category-specific weekly comparison
        with insight_cols[0]:
            st.markdown("#### Time by Category (Week-over-Week)")

            curr_by_cat = current_week.groupby('category')['duration_minutes'].sum()
            prev_by_cat = previous_week.groupby('category')['duration_minutes'].sum()

            for cat in curr_by_cat.index:
                if cat in prev_by_cat.index:
                    curr_time = curr_by_cat[cat]
                    prev_time = prev_by_cat[cat]
                    if prev_time > 0:
                        pct_change = ((curr_time - prev_time) / prev_time) * 100
                        emoji = "📈" if pct_change > 0 else "📉"
                        curr_val, unit = format_time(curr_time, time_unit)
                        prev_val, _ = format_time(prev_time, time_unit)
                        st.markdown(f"{emoji} **{cat.capitalize()}**: {abs(pct_change):.0f}% {'more' if pct_change > 0 else 'less'} time this week ({curr_val:.1f} {unit} vs {prev_val:.1f} {unit})")

        # Domain-specific weekly comparison
        with insight_cols[1]:
            st.markdown("#### Top Domain Changes")

            curr_by_domain = current_week.groupby('domain')['duration_minutes'].sum().nlargest(5)
            prev_by_domain = previous_week.groupby('domain')['duration_minutes'].sum()

            for domain in curr_by_domain.index:
                if domain in prev_by_domain.index:
                    curr_time = curr_by_domain[domain]
                    prev_time = prev_by_domain[domain]
                    if prev_time > 0:
                        pct_change = ((curr_time - prev_time) / prev_time) * 100
                        emoji = "📈" if pct_change > 0 else "📉"
                        st.markdown(f"{emoji} **{domain}**: {abs(pct_change):.0f}% {'more' if pct_change > 0 else 'less'} time")

    st.markdown("---")

    # Visualizations
    st.header("📈 Visualizations")

    # Row 1: Time series and category breakdown
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("Browsing Time Over Time")

        # Use deduplicated time if available
        if 'deduplicated_minutes' in filtered_df.columns:
            daily_time = filtered_df.groupby('date')['deduplicated_minutes'].first().reset_index()
            daily_time.columns = ['date', 'duration_minutes']
        else:
            daily_time = filtered_df.groupby('date')['duration_minutes'].sum().reset_index()

        daily_time['duration_display'] = daily_time['duration_minutes'].apply(lambda x: format_time(x, time_unit)[0])
        _, unit_label = format_time(0, time_unit)
        fig_timeline = px.line(
            daily_time,
            x='date',
            y='duration_display',
            title='Daily Browsing Time',
            labels={'duration_display': f'Time ({unit_label})', 'date': 'Date'}
        )
        fig_timeline.update_traces(
            line_color='#1f77b4',
            line_width=2,
            hovertemplate='<b>Date</b> = %{x|%b %d, %Y}<br><b>Time</b> = %{y:.1f} ' + unit_label + '<extra></extra>'
        )
        fig_timeline.update_layout(hovermode='x unified')
        st.plotly_chart(fig_timeline, width='stretch')

    with viz_col2:
        st.subheader("Time by Category")
        category_time = filtered_df.groupby('category')['duration_minutes'].sum().reset_index()
        category_time['duration_display'] = category_time['duration_minutes'].apply(lambda x: format_time(x, time_unit)[0])
        category_time = category_time.sort_values('duration_display', ascending=False)
        fig_category = px.bar(
            category_time,
            x='category',
            y='duration_display',
            title='Total Time by Category',
            labels={'duration_display': f'Time ({unit_label})', 'category': 'Category'},
            color='duration_display',
            color_continuous_scale='Blues'
        )
        fig_category.update_traces(
            hovertemplate='<b>Category</b> = %{x}<br><b>Time</b> = %{y:.1f} ' + unit_label + '<extra></extra>'
        )
        st.plotly_chart(fig_category, width='stretch')

    # Row 2: Hourly pattern and domain distribution
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.subheader("Hourly Activity Pattern")
        hourly_visits = filtered_df.groupby('hour').size().reset_index(name='visits')
        fig_hourly = px.bar(
            hourly_visits,
            x='hour',
            y='visits',
            title='Browsing Activity by Hour',
            labels={'visits': 'Number of Visits', 'hour': 'Hour of Day'},
            color='visits',
            color_continuous_scale='Viridis'
        )
        fig_hourly.update_traces(
            hovertemplate='<b>Hour</b> = %{x}:00<br><b>Visits</b> = %{y}<extra></extra>'
        )
        st.plotly_chart(fig_hourly, width='stretch')

    with viz_col4:
        st.subheader("Top 10 Domains")
        top_10_domains = filtered_df['domain'].value_counts().head(10).reset_index()
        top_10_domains.columns = ['domain', 'visits']
        fig_domains = px.pie(
            top_10_domains,
            values='visits',
            names='domain',
            title='Visit Distribution - Top 10 Domains'
        )
        fig_domains.update_traces(
            hovertemplate='<b>Domain</b> = %{label}<br><b>Visits</b> = %{value}<br><b>Percentage</b> = %{percent}<extra></extra>'
        )
        st.plotly_chart(fig_domains, width='stretch')

    # Row 3: Weekday vs Weekend and Category distribution
    viz_col5, viz_col6 = st.columns(2)

    with viz_col5:
        st.subheader("Weekday vs Weekend")
        weekend_comparison = filtered_df.groupby('is_weekend')['duration_minutes'].sum().reset_index()
        weekend_comparison['day_type'] = weekend_comparison['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        weekend_comparison['duration_display'] = weekend_comparison['duration_minutes'].apply(lambda x: format_time(x, time_unit)[0])

        # Use graph_objects for better bar width control
        fig_weekend = go.Figure()
        fig_weekend.add_trace(go.Bar(
            x=weekend_comparison['day_type'],
            y=weekend_comparison['duration_display'],
            marker=dict(color=['#636EFA', '#EF553B']),
            hovertemplate='<b>Day Type</b> = %{x}<br><b>Time</b> = %{y:.1f} ' + unit_label + '<extra></extra>',
            width=0.5
        ))
        fig_weekend.update_layout(
            title='Browsing Time: Weekday vs Weekend',
            xaxis_title='Day Type',
            yaxis_title=f'Time ({unit_label})',
            showlegend=False
        )
        st.plotly_chart(fig_weekend, width='stretch')

    with viz_col6:
        st.subheader("Category Distribution")
        category_dist = filtered_df['category'].value_counts().reset_index()
        category_dist.columns = ['category', 'visits']
        fig_cat_pie = px.pie(
            category_dist,
            values='visits',
            names='category',
            title='Visit Distribution by Category'
        )
        fig_cat_pie.update_traces(
            hovertemplate='<b>Category</b> = %{label}<br><b>Visits</b> = %{value}<br><b>Percentage</b> = %{percent}<extra></extra>'
        )
        st.plotly_chart(fig_cat_pie, width='stretch')

    # Heatmap: Day of week vs Hour
    st.subheader("Activity Heatmap: Day of Week vs Hour")
    heatmap_data = filtered_df.groupby(['day of week', 'hour']).size().reset_index(name='visits')
    heatmap_pivot = heatmap_data.pivot(index='day of week', columns='hour', values='visits').fillna(0)

    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([day for day in day_order if day in heatmap_pivot.index])

    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Visits"),
        title="Browsing Activity Heatmap",
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heatmap.update_traces(
        hovertemplate='<b>Day</b> = %{y}<br><b>Hour</b> = %{x}:00<br><b>Visits</b> = %{z:.0f}<extra></extra>'
    )
    st.plotly_chart(fig_heatmap, width='stretch')

    # Detailed data table
    st.markdown("---")
    st.header("📋 Detailed Data")

    # Show summary statistics
    with st.expander("View Summary Statistics"):
        col_stats1, col_stats2 = st.columns(2)

        with col_stats1:
            st.markdown("#### Category Stats")
            cat_stats = filtered_df.groupby('category').agg({
                'duration_minutes': 'sum',
                'url': 'count'
            }).round(2)
            cat_stats[f'Total Time ({unit_label})'] = cat_stats['duration_minutes'].apply(lambda x: format_time(x, time_unit)[0])
            cat_stats = cat_stats[[f'Total Time ({unit_label})', 'url']]
            cat_stats.columns = [f'Total Time ({unit_label})', 'Visit Count']
            cat_stats = cat_stats.sort_values(f'Total Time ({unit_label})', ascending=False)
            st.dataframe(cat_stats)

        with col_stats2:
            st.markdown("#### Top Domains Stats")
            domain_stats = filtered_df.groupby('domain').agg({
                'duration_minutes': 'sum',
                'url': 'count'
            }).round(2)
            domain_stats[f'Total Time ({unit_label})'] = domain_stats['duration_minutes'].apply(lambda x: format_time(x, time_unit)[0])
            domain_stats = domain_stats[[f'Total Time ({unit_label})', 'url']]
            domain_stats.columns = [f'Total Time ({unit_label})', 'Visit Count']
            domain_stats = domain_stats.sort_values(f'Total Time ({unit_label})', ascending=False).head(10)
            st.dataframe(domain_stats)

    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(
            filtered_df[['datetime', 'title', 'domain', 'category', 'duration_minutes', 'day of week']].sort_values('datetime', ascending=False),
            width='stretch'
        )

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit and Plotly*")
