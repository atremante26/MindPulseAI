import { 
    ComposedChart, Line, Area, XAxis, YAxis, 
    Tooltip, ResponsiveContainer
} from 'recharts'
import './ForecastChart.css'

export default function ForecastChart({ title, historical, forecast, metricName, onPointClick }) {
    const chartData = [
        ...historical.map(p => ({
            date: p.ds.split(' ')[0],
            historical: p.value,
            forecast: null,
            band: null
        })),
        ...forecast.predictions.map(p => ({
            date: p.ds.split('T')[0],
            historical: null,
            forecast: p.yhat,
            band: [p.yhat_lower, p.yhat_upper]
        }))
    ]

    return (
        <div className="forecast-chart">
            <h3 className="chart-title">{title}</h3>
            <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <XAxis 
                        dataKey="date"
                        tick={{ fill: '#8fa894', fontSize: 11 }}
                        tickFormatter={(date) => date.slice(5)}
                        interval="preserveStartEnd"
                        axisLine={{ stroke: '#1e2b21' }}
                        tickLine={false}
                    />
                    <YAxis 
                        tick={{ fill: '#8fa894', fontSize: 11 }}
                        tickFormatter={(val) => val.toFixed(1)}
                        axisLine={false}
                        tickLine={false}
                        width={45}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#1a211e',
                            border: '1px solid #1e2b21',
                            borderRadius: '8px',
                            color: '#e8f0ea',
                            fontSize: '0.85rem',
                            lineHeight: '2',        
                            padding: '0.75rem 1rem'
                        }}
                        formatter={(value, name) => {
                            if (name === 'band') {
                                if (!value) return null
                                return [`[${value[0].toFixed(2)}, ${value[1].toFixed(2)}]`, 'Confidence Interval']
                            }
                            if (name === 'forecast') return [value.toFixed(2), 'Forecast']
                            if (name === 'historical') return [value.toFixed(2), 'Value']
                            if (typeof value === 'number') return [value.toFixed(2), name]
                            return null
                        }}
                        labelFormatter={(label) => `Week of ${label}`}
                    />
                    <Area
                        dataKey="band"
                        stroke="none"
                        fill="#4a9e6b"
                        fillOpacity={0.08}
                        connectNulls={false}
                        legendType="none"
                    />
                    <Line
                        dataKey="historical"
                        stroke="#4a9e6b"
                        strokeWidth={2}
                        dot={false}
                        connectNulls={false}
                        legendType="none"
                    />
                    <Line
                        dataKey="forecast"
                        stroke="#7eb8a4"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={{ fill: '#7eb8a4', r: 3, cursor: 'pointer' }}
                        activeDot={{ 
                            r: 5, 
                            cursor: 'pointer',
                            onClick: (event, payload) => onPointClick && onPointClick(payload, metricName)
                        }}
                        connectNulls={false}
                        legendType="none"
                        onClick={(data) => onPointClick && onPointClick(data, metricName)}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    )
}