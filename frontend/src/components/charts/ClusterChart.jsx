import { 
    ScatterChart, Scatter, XAxis, YAxis, ZAxis, 
    Tooltip, Cell, ResponsiveContainer } from "recharts"
import './ClusterChart.css'

const COLORS = ['#7eb8a4', '#4a9e6b', '#c9a96e']

const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
        <div className="cluster-tooltip">
            <p className="tooltip-title">Cluster {d.id}</p>
            <p className="tooltip-stat">n = {d.size} | {d.z}% of respondents</p>
            <p className="tooltip-stat">Avg age: {d.avg_age}</p>
        </div>
    )
}

export default function ClusterChart({ clusters, onBubbleClick }) {
    const bubbleData = Object.entries(clusters.cluster_profiles).map(([id, profile]) => ({
        id: parseInt(id),
        x: parseInt(id) === 0 ? 1 : parseInt(id) === 1 ? 3 : 5,
        y: 0,
        z: profile.percentage,
        size: profile.size,
        avg_age: profile.avg_age,
        in_treatment_pct: profile.in_treatment_pct,
        remote_work_pct: profile.remote_work_pct,
        gender_distribution: profile.gender_distribution,
        top_work_interference: profile.top_work_interference,
        top_countries: profile.top_countries,
        key_traits: profile.key_traits,
        family_history_pct: profile.family_history_pct,
        benefits_distribution: profile.benefits_distribution
    }))

    return (
        <div className='cluster-chart'>
            <h3 className='cluster-title'>Topic Clusters</h3>
            <ResponsiveContainer width="100%" height={350}>
                <ScatterChart margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
                    <XAxis dataKey="x" type="number" domain={[0, 6]} hide />
                    <YAxis dataKey="y" type="number" domain={[-1, 1]} hide />
                    <ZAxis dataKey="z" range={[1000, 15000]} />
                    <Tooltip content={<CustomTooltip />} cursor={false} />
                    <Scatter
                        data={bubbleData}
                        onClick={(data) => onBubbleClick && onBubbleClick(data)}
                        style={{ cursor: 'pointer' }}
                    >
                        {bubbleData.map((entry, i) => (
                            <Cell key={i} fill={COLORS[i]} fillOpacity={0.6} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    )
}