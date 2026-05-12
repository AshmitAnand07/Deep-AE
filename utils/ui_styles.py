"""
ui_styles.py
============
Deep-AE CSS Module for Modern Zero-Gravity UI
(Styled via Stitch Design System: Deep-AE Modern AI Platform)

This module contains the custom CSS injected into the Streamlit app.
"""

def get_custom_css():
    return """
    <style>
        /* Base Global Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --stitch-bg: #050810;
            --stitch-primary: #00c9ff;
            --stitch-primary-dim: #66d3ff;
            --stitch-surface: #0e1417;
            --stitch-surface-glass: rgba(15, 23, 42, 0.6);
            --stitch-border-glass: rgba(255, 255, 255, 0.1);
            --stitch-on-surface: #dde3e8;
            --stitch-on-surface-variant: #bcc8d0;
            --stitch-error: #ffb4ab;
            --stitch-success: #66d3ff; /* Using primary-dim for success/active in this theme */
            --stitch-text-muted: #94A3B8;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--stitch-on-surface);
        }
        
        /* ----------------------------------------------------
           ZERO-GRAVITY SPACE BACKGROUND
           ---------------------------------------------------- */
        .stApp {
            background-color: var(--stitch-bg);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(0, 201, 255, 0.05), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(102, 211, 255, 0.05), transparent 25%);
            background-size: 200% 200%;
            animation: spaceDrift 30s ease-in-out infinite alternate;
            overflow-x: hidden;
        }

        .stApp::before {
            content: '';
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            pointer-events: none;
            background: 
                radial-gradient(1px 1px at 20px 30px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1.5px 1.5px at 150px 250px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 300px 80px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 450px 400px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 80px 500px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1.5px 1.5px at 600px 120px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 800px 350px, #ffffff, rgba(0,0,0,0));
            background-repeat: repeat;
            background-size: 1000px 1000px;
            animation: starlight 15s linear infinite;
            opacity: 0.3;
            z-index: 0;
        }

        @keyframes spaceDrift {
            0% { background-position: 0% 0%; }
            100% { background-position: 100% 100%; }
        }

        @keyframes starlight {
            from { transform: translateY(0); }
            to { transform: translateY(-1000px); }
        }

        /* ----------------------------------------------------
           ZERO-GRAVITY KEYFRAME ANIMATIONS
           ---------------------------------------------------- */
        @keyframes floatLeft {
            0% { transform: translateY(0px) rotate(0deg) translateX(0px); }
            33% { transform: translateY(-12px) rotate(-1deg) translateX(-5px); }
            66% { transform: translateY(8px) rotate(1deg) translateX(5px); }
            100% { transform: translateY(0px) rotate(0deg) translateX(0px); }
        }

        @keyframes floatRight {
            0% { transform: translateY(0px) rotate(0deg) translateX(0px); }
            33% { transform: translateY(15px) rotate(1.5deg) translateX(8px); }
            66% { transform: translateY(-10px) rotate(-1deg) translateX(-8px); }
            100% { transform: translateY(0px) rotate(0deg) translateX(0px); }
        }

        @keyframes floatCenter {
            0% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-15px) scale(1.02); }
            100% { transform: translateY(0px) scale(1); }
        }

        @keyframes bobbing {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
            100% { transform: translateY(0px); }
        }

        /* Typography overrides based on Stitch design */
        h1, h2, h3, h4, h5, h6 {
            color: var(--stitch-on-surface) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }

        /* ----------------------------------------------------
           UI COMPONENTS STYLED WITH STITCH TOKENS
           ---------------------------------------------------- */
        
        .glass-card {
            background: var(--stitch-surface-glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid var(--stitch-border-glass);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            padding: 24px;
            margin-bottom: 24px;
            transition: all 0.3s ease;
            position: relative;
            z-index: 10;
        }
        
        div[data-testid="column"]:nth-child(1) .glass-card {
            animation: floatLeft 12s ease-in-out infinite;
        }
        div[data-testid="column"]:nth-child(2) .glass-card {
            animation: floatRight 14s ease-in-out infinite;
        }

        .glass-card:hover {
            transform: scale(1.05) translateY(-5px) !important;
            border-color: rgba(0, 201, 255, 0.4);
            box-shadow: 0 20px 50px rgba(0, 201, 255, 0.15);
            animation-play-state: paused !important;
        }

        /* Hero Section */
        .hero-title {
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, var(--stitch-primary) 0%, var(--stitch-primary-dim) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
            animation: floatCenter 8s ease-in-out infinite;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            color: var(--stitch-text-muted);
            text-align: center;
            font-weight: 400;
            margin-bottom: 3rem;
            letter-spacing: 0.05em;
            animation: bobbing 6s ease-in-out infinite 1s;
        }

        /* Prediction Badges */
        .badge-crack {
            background: rgba(255, 180, 171, 0.15); /* Derived from stitch error token */
            color: var(--stitch-error);
            border: 1px solid rgba(255, 180, 171, 0.4);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            box-shadow: 0 0 30px rgba(255, 180, 171, 0.2);
            animation: floatCenter 4s ease-in-out infinite;
        }
        .badge-healthy {
            background: rgba(102, 211, 255, 0.15);
            color: var(--stitch-success);
            border: 1px solid rgba(102, 211, 255, 0.4);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            box-shadow: 0 0 30px rgba(102, 211, 255, 0.2);
            animation: floatCenter 5s ease-in-out infinite;
        }

        /* Custom File Uploader */
        .stFileUploader > div > div {
            background: var(--stitch-surface-glass) !important;
            border: 1px dashed var(--stitch-border-glass) !important;
            border-radius: 16px !important;
            padding: 3rem !important;
            animation: bobbing 9s ease-in-out infinite;
            transition: all 0.3s ease;
        }
        .stFileUploader > div > div:hover {
            border-color: var(--stitch-primary) !important;
            background: rgba(0, 201, 255, 0.05) !important;
            transform: scale(1.02);
            animation-play-state: paused;
        }
        
        .metric-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            border: 1px solid var(--stitch-border-glass);
            transition: all 0.3s ease;
        }
        .metric-container:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .metric-label {
            font-size: 0.875rem;
            color: var(--stitch-on-surface-variant);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.01em;
            font-weight: 500;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--stitch-on-surface);
            text-shadow: 0 0 10px rgba(255,255,255,0.2);
        }

        /* Sidebar Styling */
        .css-1d391kg, .css-1dp5vir, [data-testid="stSidebar"] { 
            background-color: var(--stitch-surface) !important;
            border-right: 1px solid var(--stitch-border-glass);
        }
        .sidebar-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--stitch-on-surface);
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--stitch-border-glass);
        }
        .sidebar-info-box {
            background: rgba(255, 255, 255, 0.03);
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 16px;
            font-size: 0.9rem;
            color: var(--stitch-on-surface-variant);
            border: 1px solid var(--stitch-border-glass);
            animation: floatRight 15s ease-in-out infinite;
        }

        /* Smooth Progress Bar */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, var(--stitch-primary), var(--stitch-primary-dim));
            box-shadow: 0 0 10px rgba(0, 201, 255, 0.5);
        }
    </style>
    """
