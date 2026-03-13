import BlurText from "../components/animations/BlurText"
import ShinyText from "../components/animations/ShinyText"
import Aurora from "../components/animations/Aurora"
import './About.css'

const MODELS = [
    {
        number: "01",
        name: "Sentiment Analysis",
        description: "VADER-based sentiment scoring applied to Reddit posts and news articles, producing daily positive/negative/neutral distributions across mental health discourse.",
        tags: ["VADER", "NLP", "Reddit API", "News API"],
    },
    {
        number: "02",
        name: "Prophet Forecasting",
        description: "Facebook Prophet time-series models trained on 90+ days of historical volume and sentiment data, generating 30-day forecasts with confidence intervals for both Reddit and news sources.",
        tags: ["Prophet", "Time Series", "Forecasting", "Confidence Intervals"],
    },
    {
        number: "03",
        name: "LLM Insights",
        description: "Claude 3.5 Sonnet interprets clicked forecast datapoints in context — surfacing statistical significance, mental health domain interpretation, and actionable recommendations grounded in WHO data.",
        tags: ["Claude 3.5 Sonnet", "Prompt Engineering", "RAG Context", "FastAPI"],
    },
    {
        number: "04",
        name: "Recommender System",
        description: "Weighted cosine similarity across concern, cost, age, and resource type feature groups. Crisis resources receive an explicit boost for ethical prioritization, with scores clamped to 0–100%.",
        tags: ["Cosine Similarity", "Feature Weighting", "Ethics-Aware", "Python"],
    },
]

const STACK = [
    { category: "Frontend", items: ["React", "Vite", "CSS", "GitHub Pages"] },
    { category: "Backend", items: ["FastAPI", "Python", "Pydantic", "Render"] },
    { category: "Infrastructure", items: ["AWS", "Snowflake", "Airflow", "Docker"] },
    { category: "ML / AI", items: ["Prophet", "Scikit-learn", "Claude API", "VADER"] },
]

const PIPELINE = [
    { step: "01", title: "Ingest", description: "Reddit posts and news articles collected daily via API, filtered for mental health relevance." },
    { step: "02", title: "Process", description: "VADER sentiment scoring applied per post/article. Aggregated into daily volume and sentiment metrics." },
    { step: "03", title: "Store", description: "Processed time-series data stored in Snowflake. Historical data powers both forecasting and trend visualization." },
    { step: "04", title: "Model", description: "Prophet trains on historical data to generate 30-day forecasts with upper/lower confidence intervals." },
    { step: "05", title: "Serve", description: "FastAPI exposes forecasts, clusters, LLM insights, and recommendations via a RESTful API deployed on Render." },
    { step: "06", title: "Visualize", description: "React frontend renders interactive charts, cluster profiles, and personalized resource recommendations." },
]

export default function About() {
    return (
        <div className="about">

            {/* Header */}
            <div className="about-header">
                <BlurText
                    text="About"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="about-title"
                />
                <ShinyText
                    text="Mental Health Analytics & Resource Discovery"
                    speed={3}
                    color="#4a9e6b"
                    shineColor="#e8f0ea"
                    className="about-label"
                />
                <p className="about-description">
                    MindPulseAI tracks and analyzes how mental health is discussed across Reddit and news media — 
                    surfacing trends, forecasting shifts in public discourse, and helping you find the right resources 
                    for your needs. Built on real data with automated weekly updates.
                </p>
            </div>

            {/* Pipeline */}
            <div className="about-section">
                <p className="about-section-label">How It Works</p>
                <h2 className="about-section-heading">Data Pipeline</h2>
                <div className="pipeline-grid">
                    {PIPELINE.map((step) => (
                        <div key={step.step} className="pipeline-card">
                            <span className="pipeline-number">{step.step}</span>
                            <h3 className="pipeline-title">{step.title}</h3>
                            <p className="pipeline-description">{step.description}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Models */}
            <div className="about-section">
                <p className="about-section-label">Under the Hood</p>
                <h2 className="about-section-heading">The Four Models</h2>
                <div className="models-grid">
                    {MODELS.map((model) => (
                        <div key={model.number} className="model-card">
                            <div className="model-card-inner">
                                <div className="model-header">
                                    <span className="model-number">{model.number}</span>
                                    <h3 className="model-name">{model.name}</h3>
                                </div>
                                <p className="model-description">{model.description}</p>
                                <div className="model-tags">
                                    {model.tags.map((tag) => (
                                        <span key={tag} className="model-tag">{tag}</span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Tech Stack */}
            <div className="about-section">
                <p className="about-section-label">Technologies</p>
                <h2 className="about-section-heading">Tech Stack</h2>
                <div className="stack-grid">
                    {STACK.map((group) => (
                        <div key={group.category} className="stack-group">
                            <p className="stack-category">{group.category}</p>
                            <div className="stack-items">
                                {group.items.map((item) => (
                                    <span key={item} className="stack-item">{item}</span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Links */}
            <div className="about-section">
                <div className="about-links-wrapper">
                    <Aurora colorStops={['#0d0f0e', '#1a3d2a', '#0d0f0e']} amplitude={1.2} blend={0.4} />
                    <div className="about-links-inner">
                        <p className="about-section-label">Explore More</p>
                        <h2 className="about-section-heading">Get Started</h2>
                        <div className="about-links">
                            <a
                                href="https://github.com/atremante26/MindPulseAI"
                                target="_blank"
                                rel="noreferrer"
                                className="about-link-card"
                            >
                                <span className="about-link-icon">⌥</span>
                                <div>
                                    <p className="about-link-title">GitHub Repository</p>
                                    <p className="about-link-sub">Source code, models, and documentation</p>
                                </div>
                                <span className="about-link-arrow">→</span>
                            </a>
                            <a
                                href="https://atremante26.github.io/MindPulseAI"
                                target="_blank"
                                rel="noreferrer"
                                className="about-link-card"
                            >
                                <span className="about-link-icon">◈</span>
                                <div>
                                    <p className="about-link-title">Live Application</p>
                                    <p className="about-link-sub">Deployed frontend on GitHub Pages</p>
                                </div>
                                <span className="about-link-arrow">→</span>
                            </a>
                            <a
                                href="https://mental-health-project-bct5.onrender.com/docs"
                                target="_blank"
                                rel="noreferrer"
                                className="about-link-card"
                            >
                                <span className="about-link-icon">◎</span>
                                <div>
                                    <p className="about-link-title">API Documentation</p>
                                    <p className="about-link-sub">FastAPI Swagger docs on Render</p>
                                </div>
                                <span className="about-link-arrow">→</span>
                            </a>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    )
}