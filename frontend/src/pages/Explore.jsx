import { useState } from "react"
import { getRecommendations } from "../services/api"
import ResourceCard from "../components/cards/ResourceCard"
import BlurText from "../components/animations/BlurText"
import ShinyText from "../components/animations/ShinyText"
import './Explore.css'

const CONCERNS = [
    { value: 'depression', label: 'Depression' },
    { value: 'anxiety', label: 'Anxiety' },
    { value: 'panic_attacks', label: 'Panic Attacks' },
    { value: 'ptsd', label: 'PTSD' },
    { value: 'adhd', label: 'ADHD' },
    { value: 'bipolar', label: 'Bipolar' },
    { value: 'eating_disorder', label: 'Eating Disorder' },
    { value: 'substance_abuse', label: 'Substance Abuse' },
    { value: 'self_harm', label: 'Self Harm' },
    { value: 'suicidal_thoughts', label: 'Suicidal Thoughts' },
    { value: 'stress', label: 'Stress' },
    { value: 'loneliness', label: 'Loneliness' },
    { value: 'relationship_issues', label: 'Relationship Issues' },
    { value: 'grief', label: 'Grief' },
    { value: 'trauma', label: 'Trauma' },
    { value: 'crisis', label: 'Crisis' },
    { value: 'schizophrenia', label: 'Schizophrenia' },
]

const RESOURCE_TYPES = [
    { value: 'app', label: 'App' },
    { value: 'therapy', label: 'Therapy' },
    { value: 'hotline', label: 'Hotline' },
    { value: 'community', label: 'Community' },
    { value: 'self_help', label: 'Self Help' },
    { value: 'medication_info', label: 'Medication Info' },
    { value: 'local_service', label: 'Local Service' },
]

const COST_TIERS = [
    { value: 'free', label: 'Free' },
    { value: 'low', label: 'Low ($0–50/mo)' },
    { value: 'medium', label: 'Medium ($50–150/mo)' },
    { value: 'high', label: 'High ($150+/mo)' },
]

export default function Explore() {
    const [concerns, setConcerns] = useState([])
    const [costPreference, setCostPreference] = useState('free')
    const [age, setAge] = useState('')
    const [resourceTypes, setResourceTypes] = useState([])
    const [onlineOnly, setOnlineOnly] = useState(false)
    const [crisisNeed, setCrisisNeed] = useState(false)
    const [recommendations, setRecommendations] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const toggleConcern = (value) => {
        setConcerns(prev =>
            prev.includes(value) ? prev.filter(c => c !== value) : [...prev, value]
        )
    }

    const toggleResourceType = (value) => {
        setResourceTypes(prev =>
            prev.includes(value) ? prev.filter(r => r !== value) : [...prev, value]
        )
    }

    const handleSubmit = async () => {
        if (concerns.length === 0) return
        setLoading(true)
        setError(null)
        try {
            const payload = {
                concerns,
                cost_preference: costPreference,
                age: parseInt(age),
                resource_type_preferences: resourceTypes.length > 0 ? resourceTypes : null,
                online_only: onlineOnly,
                crisis_need: crisisNeed 
            }
            const response = await getRecommendations(payload)
            setRecommendations(response.data.recommendations)
        } catch (err) {
            setError('Failed to get recommendations. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="explore">
            <div className="explore-header">
                <BlurText
                    text="Explore Resources"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="explore-title"
                />
                <ShinyText
                    text="Personalized Mental Health Recommendations"
                    speed={3}
                    color="#4a9e6b"
                    shineColor="#e8f0ea"
                    className="explore-label"
                />
                <p className="explore-description">
                    Answer a few questions to receive personalized mental health resource recommendations — matched to your needs, budget, and preferences.
                </p>
            </div>

            <div className="explore-quiz">
                {/* Concerns */}
                <div className="quiz-section">
                    <p className="quiz-label">What are you dealing with? <span className="quiz-required">*</span></p>
                    <div className="pill-grid">
                        {CONCERNS.map(c => (
                            <button
                                key={c.value}
                                className={`pill ${concerns.includes(c.value) ? 'pill-active' : ''}`}
                                onClick={() => toggleConcern(c.value)}
                            >
                                {c.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Cost */}
                <div className="quiz-section">
                    <p className="quiz-label">What's your budget?</p>
                    <div className="pill-row">
                        {COST_TIERS.map(c => (
                            <button
                                key={c.value}
                                className={`pill ${costPreference === c.value ? 'pill-active' : ''}`}
                                onClick={() => setCostPreference(c.value)}
                            >
                                {c.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Age */}
                <div className="quiz-section">
                    <p className="quiz-label">How old are you?</p>
                    <input
                        type="number"
                        className="age-input"
                        placeholder="Enter your age"
                        value={age}
                        onChange={e => setAge(e.target.value)}
                        min="13"
                        max="100"
                    />
                </div>

                {/* Resource Types */}
                <div className="quiz-section">
                    <p className="quiz-label">What kind of support are you looking for? <span className="quiz-optional">(optional)</span></p>
                    <div className="pill-row">
                        {RESOURCE_TYPES.map(r => (
                            <button
                                key={r.value}
                                className={`pill ${resourceTypes.includes(r.value) ? 'pill-active' : ''}`}
                                onClick={() => toggleResourceType(r.value)}
                            >
                                {r.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Toggles */}
                <div className="quiz-section quiz-toggles">
                    <label className="toggle-label">
                        <input
                            type="checkbox"
                            checked={onlineOnly}
                            onChange={e => setOnlineOnly(e.target.checked)}
                        />
                        Online resources only
                    </label>
                    <label className="toggle-label">
                        <input
                            type="checkbox"
                            checked={crisisNeed}
                            onChange={e => setCrisisNeed(e.target.checked)}
                        />
                        I need crisis support
                    </label>
                </div>

                {/* Submit */}
                <button
                    className="explore-submit"
                    onClick={handleSubmit}
                    disabled={loading || concerns.length === 0}
                >
                    {loading ? 'Finding resources...' : 'Get Recommendations'}
                </button>

                {error && <p className="explore-error">{error}</p>}
            </div>

            {/* Results */}
            {recommendations && (
                <div className="recommendations">
                    <p className="recommendations-label">
                        {recommendations.length} resources found
                    </p>
                    {recommendations.map((rec, i) => (
                        <ResourceCard key={i} rec={rec} />
                    ))}
                </div>
            )}
        </div>
    )
}