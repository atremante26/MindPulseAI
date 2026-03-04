import "./WeeklyInsightCard.css"

export default function WeeklyInsightCard({ title, content, currentIndex, total }) {
    return (
        <div className="weekly-insight-card">
            <h2 className="card-title">{title}</h2>
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