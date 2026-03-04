import { SiGithub } from '@icons-pack/react-simple-icons'
import './Footer.css'

export default function Footer() {
    return (
        <footer className='footer'>
            <div className='footer-github'>
                <a href='https://github.com/atremante26/MindPulseAI' target='_blank' rel='noreferrer'>
                    <SiGithub size={28} />
                </a>
            </div>
            <span className='footer-brand'>© 2026 MindPulseAI</span>
        </footer>
    )
}