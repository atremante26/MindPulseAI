import { NavLink } from 'react-router-dom'
import { Brain } from 'lucide-react'
import ShinyText from '../animations/ShinyText'
import './Navbar.css'

export default function Navbar() {
    return (
        <nav className='navbar'>
            <span className='navbar-brand'>
                <Brain size={35} />
                <ShinyText
                    text="MindPulseAI"
                    speed={3}
                    color="#4a9e6b"
                    ShineColor="#e8f0ea"
                />
            </span>
            <div className='navbar-links'>
                <NavLink to="/" end>Home</NavLink>
                <NavLink to="/dashboard">Dashboard</NavLink>
                <NavLink to="/explore">Explore</NavLink>
                <NavLink to="/about">About</NavLink>
            </div>
        </nav>
    )
}

