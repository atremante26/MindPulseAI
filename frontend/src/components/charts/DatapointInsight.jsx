import './DatapointInsight.css'

export default function DatapointInsight ({ insight }) {
     console.log('insight:', insight)
    if (!insight) return (
        <div className="datapoint-insight placeholder">
            <p>Click any forecast point to generate an AI insight.</p>
        </div>
    )

    if (insight === 'loading') return (
        <div className="datapoint-insight loading">
            <p>Generating insight...</p>
        </div>
    )

    if (!insight.metadata) return (
        <div className="datapoint-insight">
            <p className="datapoint-text">{insight.text}</p>
        </div>
    )
    
    const titleMapping = {
        reddit_volume: "Reddit Volume",
        reddit_sentiment: "Reddit Sentiment",
        news_volume: "News Volume",
        news_sentiment: "News Sentiment"
    }

    return (
        <div className="datapoint-insight">
            <div className="datapoint-header">
                <h3 className="datapoint-title">
                    {titleMapping[insight.metadata.metric]}
                </h3>
                <span className="datapoint-meta">
                    Week of {insight.metadata.week} · Value: {insight.metadata.value.toFixed(2)}
                </span>
            </div>
            <p className="datapoint-text">{insight.text}</p>
        </div>
    )
}