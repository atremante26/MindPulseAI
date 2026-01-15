from insights import get_static_context, get_crisis_indicators, get_common_concerns

def get_weekly_insights_prompt(
    week_start,
    week_end,
    reddit_sample,
    news_sample,
    forecast_summary
):
    """
    Generate Claude prompt for weekly mental health insights, including data from Reddit, News, and static datasets.

    Parameters:
    - week_start/week_end: Date range (YYYY-MM-DD)
    - reddit_sample: List of dicts with Reddit post data
    - news_sample: List of news headline strings
    - forecast_summary: Dict with forecast metrics
    
    Returns:
    - String prompt for Claude
    """
    
    static_context = get_static_context()
    crisis_indicators = get_crisis_indicators()
    common_concerns = get_common_concerns()
    
    prompt = f"""You are a mental health data analyst providing insights on community mental health trends. Analyze the following data from the week of {week_start} to {week_end}.

<data_sources>

<reddit_discussions>
Sample of {len(reddit_sample)} recent Reddit posts about mental health:

{format_reddit_sample(reddit_sample)}
</reddit_discussions>

<news_coverage>
Sample of {len(news_sample)} recent mental health news headlines:

{format_news_sample(news_sample)}
</news_coverage>

<forecast_trends>
Predictions for the next 13 weeks based on time-series forecasting models:

**Reddit Discussion Volume:**
- Average: {forecast_summary['reddit_volume_avg']:.0f} posts/week
- Trend: {forecast_summary['reddit_volume_trend']}
- Change: {forecast_summary['reddit_volume_change']:+.1f}%

**Reddit Community Sentiment:**
- Average: {forecast_summary['reddit_sentiment_avg']:.2f} (scale: -1 to +1)
- Trend: {forecast_summary['reddit_sentiment_trend']}
- Change: {forecast_summary['reddit_sentiment_change']:+.1f}%

**News Coverage Volume:**
- Average: {forecast_summary['news_volume_avg']:.0f} articles/week
- Trend: {forecast_summary['news_volume_trend']}
- Change: {forecast_summary['news_volume_change']:+.1f}%

**News Sentiment:**
- Average: {forecast_summary['news_sentiment_avg']:.2f}
- Trend: {forecast_summary['news_sentiment_trend']}
- Change: {forecast_summary['news_sentiment_change']:+.1f}%

**Coverage Gap Analysis:**
- Reddit/News Ratio: {forecast_summary['coverage_ratio']:.1f}x
- Interpretation: {"Community discussions far exceed media coverage" if forecast_summary['coverage_ratio'] > 1 else "Media coverage exceeds community discussions"}
</forecast_trends>

</data_sources>

<context>
Statistical summaries and insights from historic datasets including WHO Suicide, Mental Health Care, and Suicide Demographics. Consider these as useful context for understanding the mental health conversation.
{static_context}
</context>

<crisis_indicators>
List of crisis-related keywords to flag.
{crisis_indicators}
</crisis_indicators>

<common_concerns>
Common mental health concerns to look for in data. Useful for thematic analysis.
{common_concerns}
</common_concerns>

<task>
Provide a structured analysis with the following sections. Be specific, data-driven, and empathetic in your tone.

**1. KEY THEMES** (2-3 sentences)
What are the dominant topics in Reddit discussions this week? What themes are appearing in news coverage? Are they aligned or divergent?

**2. SENTIMENT ANALYSIS** (2-3 sentences)
How is the mental health community feeling based on Reddit sentiment scores? Is the forecasted trend improving or worsening? What might explain this pattern?

**3. COVERAGE GAP INSIGHTS** (2-3 sentences)
Compare the volume of Reddit discussions to news coverage. What does this gap tell us about public discourse vs. media priorities? Are important community concerns being overlooked by mainstream media?

**4. CONCERNING PATTERNS** (1-3 sentences)
Are there any crisis indicators, unusual spikes, or worrying trends in the data? If none detected, state "No immediate crisis indicators detected this week."

**5. RECOMMENDATIONS** (3-4 bullet points)
Provide actionable insights for mental health stakeholders (organizations, policymakers, support services):
- What areas need immediate attention?
- What resources should be prioritized?
- What trends should be monitored closely?
</task>

Format your response with clear section headers and concise, evidence-based insights.
"""
    
    return prompt


def format_reddit_sample(reddit_posts):
    """
    Format Reddit posts for Claude prompt.

    Parameters:
    - reddit_posts: List of dicts with keys: title, sentiment, score, comments
    
    Returns:
    - Formatted string
    """
    if not reddit_posts:
        return "No Reddit posts available for this period."
    
    formatted = []
    for i, post in enumerate(reddit_posts[:30], 1):  # Limit to 30
        title = post['title'][:150]  # Truncate long titles
        sentiment = post['sentiment']
        score = post['score']
        comments = post['comments']
        
        # Add sentiment label
        if sentiment > 0.2:
            sent_label = "positive"
        elif sentiment < -0.2:
            sent_label = "negative"
        else:
            sent_label = "neutral"
        
        formatted.append(
            f"{i}. \"{title}\" "
            f"(sentiment: {sentiment:.2f} [{sent_label}], "
            f"upvotes: {score}, comments: {comments})"
        )
    
    return "\n".join(formatted)


def format_news_sample(news_headlines):
    """
    Format news headlines for Claude prompt.

    Parameters:
    - news_headlines: List of headline strings
    
    Returns:
    - Formatted string
    """
    if not news_headlines:
        return "No news headlines available for this period."
    
    formatted = []
    for i, headline in enumerate(news_headlines[:15], 1):  # Limit to 15
        formatted.append(f"{i}. {headline}")
    
    return "\n".join(formatted)