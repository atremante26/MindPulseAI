import { BrowserRouter as Router, Routes, Route} from 'react-router-dom'
import './index.css'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Explore from './pages/Explore'
import About from './pages/About'
import Navbar from './components/layouts/Navbar'
import Footer from './components/layouts/Footer'

function App() {
  return (
    <Router>
      <Navbar />
      <main>
        <Routes>
        <Route path='/MindPulseAI' element={<Home />} />
        <Route path='/MindPulseAI/dashboard' element={<Dashboard />} />
        <Route path='/MindPulseAI/explore' element={<Explore />} />
        <Route path='/MindPulseAI/about' element={<About />} />
      </Routes>
      </main>
      <Footer />
    </Router>
  )
}

export default App