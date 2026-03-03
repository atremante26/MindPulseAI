import axios from "axios"; 

// Create Axios instance
const api = axios.create({
    baseURL: "https://mental-health-project-bct5.onrender.com/api" // Create to Render backend
})

// Clustering
export const getClusters = async () => api.get("/clustering/clusters")

// Forecasting
export const getForecasts = async () => api.get("/forecasting/forecasts")

export const getHistorical = async () => api.get("/forecasting/historical")

// Insights
export const getWeekly = async () => api.get("/insights/weekly")

export const getDatapoint = async (datapointRequest) => api.post("/insights/datapoint", datapointRequest)

// Recommendations
export const getRecommendations = async (recommendationRequest) => api.post("/recommendations/recommend", recommendationRequest)