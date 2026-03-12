import CountUp from '../animations/CountUp'
import ShinyText from '../animations/ShinyText'
import './ResourceCard.css'

const formatCost = (cost) => cost.charAt(0).toUpperCase() + cost.slice(1)

export default function ResourceCard({ rec }) {
    if (!rec) return null

    const online = rec.resource.online_only ? "Yes" : "No"
    const matchPct = Math.min(100, Math.round(rec.match_score * 100))

    return (
        <div className={`resource-card ${rec.resource.crisis_resource ? 'crisis' : ''}`}>
            <div className="resource-card-top">
                <div className="resource-card-left">
                    <div className="resource-badges">
                        <span className="badge badge-type">{rec.resource.type}</span>
                        {rec.resource.crisis_resource && (
                            <span className="badge badge-crisis">Crisis Resource</span>
                        )}
                    </div>
                    <ShinyText
                        text={rec.resource.name}
                        speed={4}
                        color="#e8f0ea"
                        shineColor="#4a9e6b"
                        className="resource-name"
                    />
                    <p className='resource-description-text'>{rec.resource.description}</p>
                </div>
                <div className="resource-match-column">
                    <div className="resource-match-score">
                        <CountUp to={matchPct} />
                        <span className="match-symbol">%</span>
                    </div>
                    <p className="match-label">match</p>
                    <div className="match-bar">
                        <div
                            className="match-bar-fill"
                            style={{ width: `${matchPct}%` }}
                        />
                    </div>
                </div>
            </div>

            <div className="resource-divider" />

            <div className="resource-card-explanation">
                <div className='resource-explanation-reasons'>
                    {rec.explanation.reasons.map((reason, i) => (
                        <span key={i} className="reason-item">
                            • {reason.replace(/_/g, ' ')}
                        </span>
                    ))}
                </div>
            </div>

            <div className="resource-divider" />

            <div className='resource-card-metadata'>
                <span className='resource-metadata-info'>
                    Cost: {formatCost(rec.resource.cost_tier)}
                </span>
                <span className='resource-metadata-info'>Online: {online}</span>
                {rec.resource.url && (
                    <a href={rec.resource.url} target="_blank" rel="noreferrer" className="resource-link">
                        Visit Website →
                    </a>
                )}
                {rec.resource.phone && (
                    <span className='resource-metadata-info'>
                        Call / Text: {rec.resource.phone}
                    </span>
                )}
            </div>
        </div>
    )
}