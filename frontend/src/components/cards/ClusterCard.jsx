import { useEffect, useRef } from 'react'
import './ClusterCard.css'

function ProgressBar({ value }) {
    return (
        <div className="stat-progress-track">
            <div
                className="stat-progress-fill"
                style={{ width: `${Math.min(value, 100)}%` }}
            />
        </div>
    )
}

function GenderPieChart({ distribution }) {
    const canvasRef = useRef(null)
    const entries = Object.entries(distribution)

    const COLORS = [
        'rgba(74, 222, 128, 0.85)',
        'rgba(74, 222, 128, 0.3)',
        'rgba(74, 222, 128, 0.55)',
    ]

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        const size = canvas.width
        const cx = size / 2
        const cy = size / 2
        const radius = size / 2 - 4

        ctx.clearRect(0, 0, size, size)

        const total = entries.reduce((sum, [, v]) => sum + v, 0)
        let startAngle = -Math.PI / 2

        entries.forEach(([, pct], i) => {
            const slice = (pct / total) * 2 * Math.PI
            ctx.beginPath()
            ctx.moveTo(cx, cy)
            ctx.arc(cx, cy, radius, startAngle, startAngle + slice)
            ctx.closePath()
            ctx.fillStyle = COLORS[i % COLORS.length]
            ctx.fill()
            startAngle += slice
        })

        // donut hole
        ctx.beginPath()
        ctx.arc(cx, cy, radius * 0.55, 0, 2 * Math.PI)
        ctx.fillStyle = '#1a2e1a'
        ctx.fill()
    }, [])

    return (
        <div className="gender-pie-wrapper">
            <canvas ref={canvasRef} width={90} height={90} className="gender-pie-canvas" />
            <div className="gender-pie-legend">
                {entries.map(([gender, pct], i) => (
                    <div key={gender} className="gender-legend-item">
                        <span
                            className="gender-legend-dot"
                            style={{ backgroundColor: COLORS[i % COLORS.length] }}
                        />
                        <span className="gender-legend-label">{gender}</span>
                        <span className="gender-legend-pct">{pct}%</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

const INTERFERENCE_COLORS = {
    never: 'var(--color-primary)',
    rarely: 'var(--color-primary)',
    sometimes: '#f59e0b',
    often: '#f97316',
    always: '#ef4444',
}

function InterferenceDot({ value }) {
    const key = value?.toLowerCase() || ''
    const color = INTERFERENCE_COLORS[key] || 'var(--color-muted)'
    return <span className="interference-dot" style={{ backgroundColor: color }} />
}

export default function ClusterCard({ cluster }) {
    if (!cluster) return <div></div>

    const hasTraits =
        cluster.key_traits &&
        (Array.isArray(cluster.key_traits)
            ? cluster.key_traits.length > 0
            : Object.keys(cluster.key_traits).length > 0)

    const traitList = hasTraits
        ? Array.isArray(cluster.key_traits)
            ? cluster.key_traits
            : Object.keys(cluster.key_traits)
        : []

    const countries = Object.keys(cluster.top_countries)

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
                    <p className="stat-sublabel">years old</p>
                </div>
                <div className="cluster-stat">
                    <p className="stat-label">In Treatment</p>
                    <p className="stat-value">{cluster.in_treatment_pct}%</p>
                    <ProgressBar value={cluster.in_treatment_pct} />
                </div>
                <div className="cluster-stat">
                    <p className="stat-label">Remote Work</p>
                    <p className="stat-value">{cluster.remote_work_pct}%</p>
                    <ProgressBar value={cluster.remote_work_pct} />
                </div>
                <div className="cluster-stat">
                    <p className="stat-label">Family History</p>
                    <p className="stat-value">{cluster.family_history_pct}%</p>
                    <ProgressBar value={cluster.family_history_pct} />
                </div>
            </div>

            {hasTraits && (
                <div className="cluster-section">
                    <p className="cluster-section-label">Key Traits</p>
                    <div className="cluster-traits">
                        {traitList.map((trait, i) => (
                            <span key={i} className="trait-pill">{trait}</span>
                        ))}
                    </div>
                </div>
            )}

            <div className="cluster-bottom-grid">
                <div className="cluster-bottom-section">
                    <p className="cluster-section-label">Gender Distribution</p>
                    <GenderPieChart distribution={cluster.gender_distribution} />
                </div>

                <div className="cluster-bottom-section">
                    <p className="cluster-section-label">Work Interference</p>
                    <div className="interference-value">
                        <InterferenceDot value={cluster.top_work_interference} />
                        <p className="cluster-section-value">{cluster.top_work_interference}</p>
                    </div>
                </div>

                <div className="cluster-bottom-section">
                    <p className="cluster-section-label">Top Countries</p>
                    <div className="countries-list">
                        {countries.map((country, i) => (
                            <p key={i} className="country-item">{country}</p>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}