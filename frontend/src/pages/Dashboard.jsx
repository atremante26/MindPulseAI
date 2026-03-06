import { useState, useEffect } from "react"
import { getForecasts, getHistorical, getDatapoint, getClusters } from "../services/api"
import ForecastChart from "../components/charts/ForecastChart"
import BlurText from "../components/animations/BlurText"
import ShinyText from "../components/animations/ShinyText"
import './Dashboard.css'

export default function Dashboard() {
    const [forecasts, setForecasts] = useState(null)
    const [historical, setHistorical] = useState(null)
    const [datapoint, setDatapoint] = useState(null)
    const [clusters, setClusters] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

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
                console.log(forecastsResponse.data.news_volume.predictions)
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

    const handlePointClick = (data, metricName) => {
        console.log('clicked:', data, metricName)
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
                        metricName="volume"
                        onPointClick={handlePointClick}
                    />
                </div>
                <div className="chart-with-insight">
                    <ForecastChart
                        title="News Volume"
                        historical={historical.news_volume}
                        forecast={forecasts.news_volume}
                        metricName="volume"
                        onPointClick={handlePointClick}
                    />
                </div>
                <div className="chart-with-insight">
                    <ForecastChart
                        title="Reddit Sentiment"
                        historical={historical.reddit_sentiment}
                        forecast={forecasts.reddit_sentiment}
                        metricName="reddit_sentiment"
                        onPointClick={handlePointClick}
                    />
                </div>
                <div className="chart-with-insight">
                    <ForecastChart
                        title="News Sentiment"
                        historical={historical.news_sentiment}
                        forecast={forecasts.news_sentiment}
                        metricName="news_sentiment"
                        onPointClick={handlePointClick}
                    />
                </div>
            </div>
        </div>
    )
}