import ShinyText from "../animations/ShinyText"
import "./WeeklyInsightCard.css"

export default function WeeklyInsightCard({ title, content, currentIndex, total }) {
    return (
        <div className="weekly-insight-card">
            <h2 className="card-title">
                <ShinyText
                    text={title}
                    speed={2}
                    color="#4a9e6b"
                    shineColor="#e8f0ea"
                    className="card-title"
                />
            </h2>
            <p className="card-content">{content}</p>
            <div className="card-dots">
                {Array.from({ length: total }).map((_, i) => (
                    <div 
                        key={i}
                        className={`card-dot ${i === currentIndex ? 'active' : ''}`}
                    />
                ))}
            </div>
        </div>
    )
}