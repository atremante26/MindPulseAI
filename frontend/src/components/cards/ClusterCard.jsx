import './ClusterCard.css'

export default function ClusterCard({ cluster }) {
    if (!cluster) return <div></div>

    return (
        <div className='cluster-card'>
            <div className="cluster-card-header">
                <h3 className='cluster-card-title'>Cluster {cluster.id}</h3>
                <span className="cluster-card-meta">{cluster.z}% of respondents</span>
            </div>
            <div className="cluster-stats-grid">
                <div className="cluster-stat">
                    <p className="stat-label">Avg Age</p>
                    <p className="stat-value">{cluster.avg_age}</p>
                </div>
                <div className="cluster-stat">
                    <p className="stat-label">In Treatment</p>
                    <p className="stat-value">{cluster.in_treatment_pct}%</p>
                </div>
                <div className="cluster-stat">
                    <p className="stat-label">Remote Work</p>
                    <p className="stat-value">{cluster.remote_work_pct}%</p>
                </div>
                <div className="cluster-stat">
                    <p className="stat-label">Family History</p>
                    <p className="stat-value">{cluster.family_history_pct}%</p>
                </div>
            </div>
            <div className="cluster-section">
                <p className="cluster-section-label">Key Traits</p>
                <div className="cluster-traits">
                    {(Array.isArray(cluster.key_traits)
                        ? cluster.key_traits
                        : Object.keys(cluster.key_traits)
                    ).map((trait, i) => (
                        <span key={i} className="trait-pill">{trait}</span>
                    ))}
                </div>
            </div>
            <div className="cluster-section">
                <p className="cluster-section-label">Gender Distribution</p>
                <div className="cluster-gender">
                    {Object.entries(cluster.gender_distribution).map(([gender, pct], i) => (
                        <span key={i} className="gender-item">{gender}: {pct}%</span>
                    ))}
                </div>
            </div>
            <div className="cluster-two-col">
                <div className="cluster-section">
                    <p className="cluster-section-label">Work Interference</p>
                    <p className="cluster-section-value">{cluster.top_work_interference}</p>
                </div>
                <div className="cluster-section">
                    <p className="cluster-section-label">Top Countries</p>
                    <p className="cluster-section-value">
                        {Object.keys(cluster.top_countries).join(' · ')}
                    </p>
                </div>
            </div>
        </div>
    )
}