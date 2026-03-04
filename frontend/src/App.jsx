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
    <Router basename='/MindPulseAI'>
      <Navbar />
      <main>
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/dashboard' element={<Dashboard />} />
          <Route path='/explore' element={<Explore />} />
          <Route path='/about' element={<About />} />
      </Routes>
      </main>
      <Footer />
    </Router>
  )
}

export default App