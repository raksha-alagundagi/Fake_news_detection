import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import base64
import os

# Try downloading NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page Configuration
st.set_page_config(page_title="Fake News Detection System", page_icon="📰", layout="centered", initial_sidebar_state="collapsed")

# Hide Streamlit's default header and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session states
if 'splash_done' not in st.session_state:
    st.session_state.splash_done = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# ----------------- GLOBAL CSS -----------------
st.markdown("""
    <style>
    /* Global Background - Solid Black */
    .stApp {
        background-color: #000000 !important;
        background-image: none !important;
        color: #ffffff !important;
    }
    
    /* Custom Header */
    .custom-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: rgba(8, 11, 18, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 99999;
        padding: 1rem 2.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 30px;
    }
    .header-logo {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .nav-links {
        display: flex;
        gap: 25px;
    }
    .nav-links a {
        color: #8b949e;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
        transition: color 0.3s;
    }
    .nav-links a:hover {
        color: #58a6ff;
    }
    
    .user-profile {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        padding: 6px 16px 6px 6px;
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: background 0.3s;
    }
    .user-profile:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    .avatar {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #a371f7, #58a6ff);
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
        font-size: 1rem;
    }
    .user-profile span {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Custom Footer */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(8, 11, 18, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 99999;
        padding: 1.2rem 2.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #8b949e;
        font-size: 0.85rem;
    }
    .footer-links {
        display: flex;
        gap: 20px;
    }
    .footer-links a {
        color: #8b949e;
        text-decoration: none;
        transition: color 0.3s;
    }
    .footer-links a:hover {
        color: #58a6ff;
    }

    /* Text Area & Input Styling */
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1.05rem !important;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stTextArea"] textarea:focus, div[data-testid="stTextInput"] input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2), inset 0 2px 5px rgba(0, 0, 0, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Login Form Container Box */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 3rem !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Button Styling */
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 15px rgba(35, 134, 54, 0.2) !important;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #3fb950 0%, #2ea043 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(35, 134, 54, 0.4) !important;
    }

    /* Medium/Short Result Cards */
    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        animation: scaleIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        max-width: 480px;
        margin-left: auto;
        margin-right: auto;
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    .fake-news {
        background: linear-gradient(135deg, #d73a49 0%, #a71d2a 100%);
        border: 1px solid rgba(255, 123, 114, 0.4);
    }
    .real-news {
        background: linear-gradient(135deg, #238636 0%, #165c26 100%);
        border: 1px solid rgba(63, 185, 80, 0.4);
    }
    .result-card h2 {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 0.4rem;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .result-card p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin: 0;
        font-weight: 600;
    }
    
    .login-title {
        text-align: center;
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- RENDER HEADER & FOOTER -----------------
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def render_header_footer():
    # Only render this if logged in
    initial = st.session_state.username[0].upper() if st.session_state.username else "U"
    user_html = f'<div class="user-profile"><div class="avatar">{initial}</div><span>{st.session_state.username}</span></div>'
    
    header_html = f"""
<div class="custom-header">
    <div class="header-left">
        <h2 class="header-logo">📰 AI News Guard</h2>
        <div class="nav-links">
            <a href="#">Home</a>
            <a href="#">Dashboard</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        {user_html}
    </div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)

    # Footer
    footer_html = """
<div class="custom-footer">
    <div>&copy; 2026 <b>AI News Guard</b> • All Rights Reserved</div>
    <div class="footer-links">
        <a href="#">Privacy Policy</a>
        <a href="#">Terms of Service</a>
        <a href="#">API Documentation</a>
    </div>
</div>
"""
    st.markdown(footer_html, unsafe_allow_html=True)

# ----------------- LOGIC -----------------
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        return None, None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    words = [ps.stem(word) for word in words if not word in set(all_stopwords)]
    return ' '.join(words)

def main():
    # 1. SPLASH SCREEN
    if not st.session_state.splash_done:
        splash_placeholder = st.empty()
        with splash_placeholder.container():
            st.markdown("""
                <style>
                .splash-container {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 80vh;
                }
                .zoom-title {
                    font-family: 'Inter', sans-serif;
                    font-size: 5rem;
                    font-weight: 900;
                    color: #ffffff;
                    text-align: center;
                    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: zoomIn 2.2s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
                    opacity: 0;
                }
                @keyframes zoomIn {
                    0% { transform: scale(0.1); opacity: 0; filter: blur(10px); }
                    60% { transform: scale(1.1); opacity: 1; filter: blur(0px); }
                    100% { transform: scale(1); opacity: 1; filter: blur(0px); }
                }
                </style>
                <div class="splash-container">
                    <h1 class="zoom-title">Fake News Detection</h1>
                </div>
            """, unsafe_allow_html=True)
        time.sleep(2.5)
        splash_placeholder.empty()
        st.session_state.splash_done = True
        st.rerun()
        return

    # 2. LOGIN SCREEN (No Header/Footer)
    if not st.session_state.logged_in:
        # Spacer for perfect vertical centering
        st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="login-title">Welcome Back</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; color:#8b949e; margin-bottom: 2.5rem;">Log in to access the AI News Guard system.</p>', unsafe_allow_html=True)
        
        # Center the form
        col1, col2, col3 = st.columns([1, 2.2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("<h3 style='text-align: center; color: white; margin-bottom: 1rem;'>Secure Portal</h3>", unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Sign In")
                
                if submitted:
                    if username.strip() != "":
                        st.session_state.username = username.strip()
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("Please enter a valid username.")
        return

    # 3. MAIN APP (Header/Footer Only Appear Here)
    render_header_footer()
    
    st.markdown("<h1 style='text-align:center; color:white; font-size: 2.5rem; margin-top:2rem;'>Analyze News Article</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8b949e; margin-bottom:2rem;'>Paste the content below to verify its authenticity using our NLP model.</p>", unsafe_allow_html=True)

    model, vectorizer = load_model_and_vectorizer()

    if model and vectorizer:
        user_input = st.text_area("", height=180, placeholder="E.g., Breaking news: Scientists discover a new planet...")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Authenticity"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                with st.spinner('Applying NLP text vectorization...'):
                    # Preprocess
                    cleaned_input = clean_text(user_input)
                    
                    if cleaned_input.strip() == "":
                         st.warning("⚠️ The input text does not contain enough valid words for analysis.")
                    else:
                        # Vectorize
                        vectorized_input = vectorizer.transform([cleaned_input])
                        
                        # Predict
                        prediction = model.predict(vectorized_input)[0]
                        probabilities = model.predict_proba(vectorized_input)[0]
                        
                        confidence = probabilities[1] if prediction == 1 else probabilities[0]
                        confidence_percentage = round(confidence * 100, 2)
                        
                        if prediction == 1:
                            st.markdown(f"""
                                <div class="result-card fake-news">
                                    <h2>🚨 FAKE NEWS DETECTED</h2>
                                    <p>Confidence Score: <b>{confidence_percentage}%</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="result-card real-news">
                                    <h2>✅ LIKELY REAL NEWS</h2>
                                    <p>Confidence Score: <b>{confidence_percentage}%</b></p>
                                </div>
                            """, unsafe_allow_html=True)
    else:
        st.error("Model files not found. Please run `python train.py` first.")

if __name__ == '__main__':
    main()
