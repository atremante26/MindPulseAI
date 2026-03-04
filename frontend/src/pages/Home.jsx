import { useState, useEffect } from 'react'
import { getWeekly } from '../services/api'
import WeeklyInsightCard from '../components/cards/WeeklyInsightCard'

export default function Home() {

    const [insights, setInsights] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [currentIndex, setCurrentIndex] = useState(0) // which insight is showing

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await getWeekly()
                console.log(response.data)  
                setInsights(response.data)
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

    // Render page with data
    return (
        <div>
            <WeeklyInsightCard 
                title={currentSection?.title}
                content={currentSection?.content}
                currentIndex={currentIndex}
                total={sections.length}
            />
        </div>
    )
}