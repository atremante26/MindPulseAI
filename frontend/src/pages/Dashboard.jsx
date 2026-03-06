import { useState, useEffect } from "react"
import { getForecasts, getHistorical, getDatapoint, getClusters } from "../services/api"
import ForecastChart from "../components/charts/ForecastChart"
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