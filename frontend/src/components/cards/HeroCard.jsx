import "./HeroCard.css"

export default function HeroCard({ weekStart, weekEnd, generatedAt, totalRedditPosts, totalNewsArticles, weeksOfData }) {
    const formatDate = (raw) => {
        if (!raw) return ''
        const date = raw.split('_')[0] // no timestamp
        return `${date.slice(0,4)}-${date.slice(4,6)}-${date.slice(6,8)}`
    }

    return (
        <div className="hero-card">
            <h2 className="hero-title">Weekly Stats</h2>
            <p className="hero-dates">
                Current Week: {weekStart} to {weekEnd}
            </p>
            <p className="hero-generated">Last Updated: {formatDate(generatedAt)}</p>
            <div className="hero-stats">
                <div className="hero-stat">
                    <span className="hero-stat-value">{totalRedditPosts}</span>
                    <span className="hero-stat-label">Reddit Posts Analyzed</span>
                </div>
                <div className="hero-stat">
                    <span className="hero-stat-value">{totalNewsArticles}</span>
                    <span className="hero-stat-label">News Articles Analyzed</span>
                </div>
                <div className="hero-stat">
                    <span className="hero-stat-value">{weeksOfData}</span>
                    <span className="hero-stat-label">Weeks of Data</span>
                </div>
            </div>
        </div>
    )
}