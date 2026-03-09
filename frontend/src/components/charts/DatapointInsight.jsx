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

    const formatInsightText = (text) => {
        const sectionLabels = [
            'Statistical Context',
            'Mental Health Interpretation',
            'Significance & Recommendations'
        ]

        // strip <response> tags if present
        let result = text.replace(/<\/?response>/g, '').trim()

        // handle all three formats:
        // 1. **Label** (no colon)
        // 2. Label: (with colon, same line)
        // 3. Label:\n (with colon, newline after)
        sectionLabels.forEach(label => {
            result = result
                .replace(`**${label}**`, `||${label}:`)
                .replace(new RegExp(`${label}:\\s*\\n`, 'g'), `||${label}:\n`)
                .replace(new RegExp(`${label}:(?!\\s*\\n)`, 'g'), `||${label}:`)
        })

        return result.split('||').filter(s => s.trim())
    }
    const sections = formatInsightText(insight.text)

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
            <div className="datapoint-sections">
                {sections.map((section, i) => {
                    const colonIdx = section.indexOf(':')
                    const label = section.slice(0, colonIdx + 1)
                    const content = section.slice(colonIdx + 1).trim()
                    return (
                        <div key={i} className="datapoint-section">
                            <span className="section-header">{label}</span>
                            <p className="datapoint-text">{content}</p>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}