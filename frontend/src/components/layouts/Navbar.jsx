import { NavLink } from 'react-router-dom'
import { Brain } from 'lucide-react'
import './Navbar.css'

export default function Navbar() {
    return (
        <nav className='navbar'>
            <span className='navbar-brand'>
                <Brain size={35} />
                MindPulseAI
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

// TODO: Add ECG-pulse thru / under the MindPulse AI text