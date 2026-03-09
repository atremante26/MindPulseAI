import { useState, useEffect } from "react"
import { getForecasts, getHistorical, getDatapoint, getClusters } from "../services/api"
import ForecastChart from "../components/charts/ForecastChart"
import DatapointInsight from "../components/charts/DatapointInsight"
import ClusterChart from "../components/charts/ClusterChart"
import BlurText from "../components/animations/BlurText"
import ShinyText from "../components/animations/ShinyText"
import Aurora from "../components/animations/Aurora"
import './Dashboard.css'

export default function Dashboard() {
    const [forecasts, setForecasts] = useState(null)
    const [historical, setHistorical] = useState(null)
    const [clusters, setClusters] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [datapointInsights, setDatapointInsights] = useState({
        reddit_volume: null,
        reddit_sentiment: null,
        news_volume: null,
        news_sentiment: null
    })
    const [selectedCluster, setSelectedCluster] = useState(null)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [forecastsResponse, historicalResponse, clustersResponse] = await Promise.all([
                    getForecasts(),
                    getHistorical(),
                    getClusters()
                ])
                setForecasts(forecastsResponse.data)
                setHistorical(historicalResponse.data)
                setClusters(clustersResponse.data)
                // console.log(forecastsResponse.data.news_volume.predictions)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    // Handle loading state
    if (loading) return <div>Loading...</div>

    // Handle error state
    if (error) return <div>Error: {error}</div>

    const handlePointClick = async (data, metricName) => {
        const clickedPoint = data.payload

        setDatapointInsights(prev => ({ ...prev, [metricName]: 'loading' }))

        const historicalData = historical[metricName]
        const clickedDate = clickedPoint.date

        const idx = historicalData.findIndex(p => p.ds.split(' ')[0] === clickedDate)
        const surrounding = historicalData
            .slice(Math.max(0, idx - 3), idx + 4)
            .filter(p => p.ds.split(' ')[0] !== clickedDate)
            .map(p => ({ date: p.ds.split(' ')[0], value: p.value }))

        const forecastData = forecasts[metricName]
        const baseline = forecastData.predictions.reduce(
            (sum, p) => sum + p.yhat, 0
        ) / forecastData.predictions.length

        const payload = {
            metric_name: metricName,
            week_date: clickedDate,
            value: clickedPoint.forecast ?? clickedPoint.historical,
            baseline: parseFloat(baseline.toFixed(2)),
            confidence_lower: clickedPoint.band?.[0] ?? 0,
            confidence_upper: clickedPoint.band?.[1] ?? 0,
            surrounding_weeks: surrounding
        }

        console.log('payload:', payload)

        try {
            const response = await getDatapoint(payload)
            setDatapointInsights(prev => ({ 
                ...prev, 
                [metricName]: response.data 
            }))
        } catch (err) {
            setDatapointInsights(prev => ({ 
                ...prev, 
                [metricName]: { text: 'Failed to generate insight.', metadata: null } 
            }))
        }
    }

    const handleBubbleClick = (data) => {
        setSelectedCluster(data)
    }

    return (
        <div className="dashboard">
            <div className="dashboard-header">
                <BlurText
                    text="Analytics Dashboard"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="dashboard-title"
                />
                <ShinyText
                    text="Real-Time Analytics"
                    speed={3}
                    color="#4a9e6b"
                    shineColor="#e8f0ea"
                    className="dashboard-label"
                />
                <p className="dashboard-description">
                    Explore how mental health discourse is trending across Reddit and news media. Interact with forecast datapoints for AI-generated insights, and uncover patterns in workplace mental health survey clusters.
                </p>
            </div>
            <div className="forecast-grid">
                <div className="chart-with-insight">
                    <ForecastChart
                        title="Reddit Volume"
                        historical={historical.reddit_volume}
                        forecast={forecasts.reddit_volume}
                        metricName="reddit_volume"
                        onPointClick={handlePointClick}
                    />
                    <DatapointInsight insight={datapointInsights.reddit_volume} />
                </div>
                <div className="chart-with-insight">
                    <ForecastChart
                        title="News Volume"
                        historical={historical.news_volume}
                        forecast={forecasts.news_volume}
                        metricName="news_volume"
                        onPointClick={handlePointClick}
                    />
                    <DatapointInsight insight={datapointInsights.news_volume} />
                </div>
                <div className="chart-with-insight">
                    <ForecastChart
                        title="Reddit Sentiment"
                        historical={historical.reddit_sentiment}
                        forecast={forecasts.reddit_sentiment}
                        metricName="reddit_sentiment"
                        onPointClick={handlePointClick}
                    />
                    <DatapointInsight insight={datapointInsights.reddit_sentiment} />
                </div>
                <div className="chart-with-insight">
                    <ForecastChart
                        title="News Sentiment"
                        historical={historical.news_sentiment}
                        forecast={forecasts.news_sentiment}
                        metricName="news_sentiment"
                        onPointClick={handlePointClick}
                    />
                    <DatapointInsight insight={datapointInsights.news_sentiment} />
                </div>
            </div>
            <div className="cluster-section">
                <div className="cluster-header">
                    <ShinyText
                        text="Workplace Mental Health Clusters"
                        speed={3}
                        color="#4a9e6b"
                        shineColor="#e8f0ea"
                        className="cluster-heading"
                    />
                    <p className="cluster-description">
                        HDBSCAN clustering of 914 Open Sourcing Mental Illness (OSMI) survey respondents. Click a bubble to explore the full cluster profile.
                    </p>
                </div>
                <div className="cluster-wrapper">
                    <Aurora colorStops={['#0d0f0e', '#1a3d2a', '#0d0f0e']} amplitude={1.2} blend={0.4} />                    
                    <div className="cluster-inner">
                        <ClusterChart clusters={clusters} onBubbleClick={handleBubbleClick} />
                    </div>
                </div>
            </div>
        </div>
    )
}