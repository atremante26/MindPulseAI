import BlurText from "../animations/BlurText"
import CountUp from "../animations/CountUp"
import ShinyText from "../animations/ShinyText"
import "./HeroCard.css"

export default function HeroCard({ weekStart, weekEnd, generatedAt, totalRedditPosts, totalNewsArticles, weeksOfData }) {
    const formatDate = (raw) => {
        if (!raw) return ''
        const date = raw.split('_')[0] // no timestamp
        return `${date.slice(0,4)}-${date.slice(4,6)}-${date.slice(6,8)}`
    }

    return (
        <div className="hero-card">
            <div className="hero-title-wrapper">
                <BlurText 
                    text="The"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="hero-title"
                />
                <ShinyText
                    text="Pulse"
                    speed={3}
                    color="#4a9e6b"
                    shineColor="#e8f0ea"
                    className="hero-title"
                />
                <BlurText 
                    text="of Mental Health"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="hero-title"
                />
            </div>
            <p className="hero-description">
                Tracking Reddit discussions and news coverage to surface mental health trends, sentiment shifts, and AI-generated insights — refreshed every week.
            </p>
            <div className="hero-divider" />
            <p className="hero-dates">
                Current Week: {weekStart} to {weekEnd}
            </p>
            <p className="hero-generated">Last Updated: {formatDate(generatedAt)}</p>
            <div className="hero-stats">
                <div className="hero-stat">
                    <span className="hero-stat-value"> <CountUp to={totalRedditPosts}/></span>
                    <span className="hero-stat-label">Reddit Posts Analyzed</span>
                </div>
                <div className="hero-stat">
                    <span className="hero-stat-value"> <CountUp to={totalNewsArticles} /></span>
                    <span className="hero-stat-label">News Articles Analyzed</span>
                </div>
                <div className="hero-stat">
                    <span className="hero-stat-value"> <CountUp to={weeksOfData} /></span>
                    <span className="hero-stat-label">Weeks of Data</span>
                </div>
            </div>
        </div>
    )
}