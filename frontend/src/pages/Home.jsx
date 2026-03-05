import { useState, useEffect } from 'react'
import { getWeekly, getHistorical } from '../services/api'
import { useNavigate } from 'react-router-dom'
import { BarChart2, ClipboardList } from 'lucide-react'
import WeeklyInsightCard from '../components/cards/WeeklyInsightCard'
import HeroCard from '../components/cards/HeroCard'
import GlanceCard from '../components/cards/GlanceCard'
import Aurora from '../components/animations/Aurora'
import './Home.css'

export default function Home() {

    const [insights, setInsights] = useState(null)
    const [historical, setHistorical] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [currentIndex, setCurrentIndex] = useState(0) // which insight is showing
    const navigate = useNavigate()

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [weeklyResponse, historicalResponse] = await Promise.all([
                    getWeekly(),
                    getHistorical()
                ])
                // console.log(weeklyResponse.data) 
                // console.log(historicalResponse.data) 
                setInsights(weeklyResponse.data)
                setHistorical(historicalResponse.data)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    useEffect(() => {
        const timer = setInterval(() => {
            setCurrentIndex(prev => (prev + 1) % 4)
        }, 5000)  // every 5 seconds
        return () => clearInterval(timer)  // cleanup when component unmounts
    }, [currentIndex])

    const sections = [
        { title: 'Key Themes', content: insights?.sections?.key_themes },
        { title: 'Sentiment', content: insights?.sections?.sentiment_analysis },
        { title: 'Concerning Patterns', content: insights?.sections?.concerning_patterns },
        { title: 'Recommendations', content: insights?.sections?.recommendations },
    ]
    const currentSection = sections[currentIndex]

    // Handle loading state
    if (loading) return <div>Loading...</div>

    // Handle error state
    if (error) return <div>Error: {error}</div>

    const totalRedditPosts = Math.round(
        historical.reddit_volume.reduce((sum, item) => sum + item.value, 0)
    )

    const totalNewsArticles = Math.round(
        historical.news_volume.reduce((sum, item) => sum + item.value, 0)
    )

    const weeksOfData = Math.max(
        historical.reddit_volume.length,
        historical.news_volume.length
    )

    const glanceMetrics = [
        {
            label: "Sentiment Score",
            value: insights.data_summary.forecast_summary.reddit_sentiment_avg.toFixed(2),
            trend: insights.data_summary.forecast_summary.reddit_sentiment_trend,
            source: "Reddit"
        },
        {
            label: "Sentiment Score",
            value: insights.data_summary.forecast_summary.news_sentiment_avg.toFixed(2),
            trend: insights.data_summary.forecast_summary.news_sentiment_trend,
            source: "News"
        },
        {
            label: "Volume Avg",
            value: insights.data_summary.forecast_summary.reddit_volume_avg.toFixed(0),
            trend: insights.data_summary.forecast_summary.reddit_volume_trend,
            source: "Reddit"
        },
        {
            label: "Volume Avg",
            value: insights.data_summary.forecast_summary.news_volume_avg.toFixed(0),
            trend: insights.data_summary.forecast_summary.news_volume_trend,
            source: "News"
        },
        {
            label: "Sentiment Change",
            value: `${insights.data_summary.forecast_summary.reddit_sentiment_change.toFixed(1)}%`,
            trend: insights.data_summary.forecast_summary.reddit_sentiment_trend,
            source: "Reddit"
        },
        {
            label: "Coverage Ratio",
            value: insights.data_summary.forecast_summary.coverage_ratio.toFixed(2),
            trend: insights.data_summary.forecast_summary.news_volume_trend,
            source: "News"
        }
    ]

    // Render page with data
    return (
        <div className="home">
            <HeroCard
                weekStart={insights.week_start}
                weekEnd={insights.week_end}
                generatedAt={insights.generated_at}
                totalRedditPosts={totalRedditPosts}
                totalNewsArticles={totalNewsArticles}
                weeksOfData={weeksOfData}
            />
            <section className="middle-section">
                <div className="glance-section">
                    <h2 className="section-label">This Week at a Glance</h2>
                    <div className="glance-wrapper">
                        <Aurora
                            colorStops={['#0d0f0e', '#1e2b21', '#0d0f0e']}
                            amplitude={0.8}
                            blend={0.5}
                        />
                        <div className="glance-grid">
                            {glanceMetrics.map((metric, i) => (
                                <GlanceCard key={i} {...metric} />
                            ))}
                        </div>
                    </div>
                </div>
                <div className="insights-section">
                    <h2 className="section-label">This Week's Insights</h2>
                    <div className='insights-wrapper'>
                        <Aurora
                            colorStops={['#0d0f0e', '#1e2b21', '#0d0f0e']}
                            amplitude={0.8}
                            blend={0.5}
                        />
                        <div className='insights-content'>
                            <WeeklyInsightCard 
                                title={currentSection?.title}
                                content={currentSection?.content}
                                currentIndex={currentIndex}
                                total={sections.length}
                            />
                        </div>
                    </div>
                </div>
            </section>
            <section className="nav-teasers">
                <div className="nav-teaser" onClick={() => navigate('/dashboard/')}>
                    <Aurora
                        colorStops={['#0d0f0e', '#1e2b21', '#0d0f0e']}
                        amplitude={0.6}
                        blend={0.4}
                    />
                    <div className='nav-teaser-content'>
                        <BarChart2 size={40} />
                        <h3>Dashboard →</h3>
                        <p>Explore forecasts, sentiment trends, and topic clusters!</p>
                    </div>
                </div>
                <div className="nav-teaser"onClick={() => navigate('/explore/')}>
                    <Aurora
                        colorStops={['#0d0f0e', '#1e2b21', '#0d0f0e']}
                        amplitude={0.6}
                        blend={0.4}
                    />
                    <div className='nav-teaser-content'>
                        <ClipboardList size={40} />
                        <h3>Mood Explorer →</h3>
                        <p>Find personalized mental health resources by completing a short quiz!</p>
                    </div>
                </div>
            </section>
        </div>
    )
}