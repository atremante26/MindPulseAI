import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { SiReddit, SiRss } from '@icons-pack/react-simple-icons'
import './GlanceCard.css'

export default function GlanceCard({ label, value, trend, source }) {
    const isReddit = source === 'Reddit'

    const formatValue = (value) => {
        const num = parseFloat(value)
        if (!isNaN(num) && num > 0) return `+${value}`
        return value
    }

    const getTrendStyle = (trend) => {
        const positive = ['increasing', 'improving']
        const negative = ['declining', 'decreasing']
        if (positive.includes(trend)) return 'positive'
        if (negative.includes(trend)) return 'negative'
        return 'stable'
    }

    const getTrendIcon = (trend) => {
        const positive = ['increasing', 'improving']
        const negative = ['declining', 'decreasing']
        if (positive.includes(trend)) return <TrendingUp size={14} />
        if (negative.includes(trend)) return <TrendingDown size={14} />
        return <Minus size={14} />  // stable
    }
    
    return (
        <div className="glance-card">
            <span className="glance-source">
                {isReddit ? <SiReddit size={14} /> : <SiRss size={14} />}
                {source}
            </span>
            <p className="glance-label">{label}</p>
            <h2 className="glance-value">{formatValue(value)}</h2>
            <div className={`glance-trend ${getTrendStyle(trend)}`}>
                {getTrendIcon(trend)}
                <span>{trend}</span>
            </div>
        </div>
    )
}