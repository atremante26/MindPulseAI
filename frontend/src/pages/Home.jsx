import { useState, useEffect } from 'react'
import { getWeekly, getHistorical } from '../services/api'
import { useNavigate } from 'react-router-dom'
import WeeklyInsightCard from '../components/cards/WeeklyInsightCard'
import HeroCard from '../components/cards/HeroCard'
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
            <WeeklyInsightCard 
                title={currentSection?.title}
                content={currentSection?.content}
                currentIndex={currentIndex}
                total={sections.length}
            />
            <section className="nav-teasers">
                <div className="nav-teaser" onClick={() => navigate('/dashboard/')}>
                    <h3>Dashboard →</h3>
                    <p>Explore forecasts, sentiment trends, and topic clusters</p>
                </div>
                <div className="nav-teaser"onClick={() => navigate('/explore/')}>
                    <h3>Mood Explorer →</h3>
                    <p>Find personalized mental health resources</p>
                </div>

            </section>
        </div>
    )
}