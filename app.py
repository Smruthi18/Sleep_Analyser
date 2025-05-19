import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load your trained model
model = joblib.load("sleep_model.pkl")

st.set_page_config(page_title="Sleep Analyzer", layout="centered")

# Global styles
st.markdown(
    """
    <style>
    .stApp {
        background-color: #A796E8;
        background-size: cover;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #080942 !important;
        font-weight: 600;
    }
    .stPlotlyChart iframe {
        border-radius: 6px !important;
        overflow: hidden;
        background-color: #431C76 !important;
    }
    .blue-text {
        color: #191970 !important;
        font-size: 20px !important;
        font-weight: 600;
    }
    section[data-testid="stFileUploader"] div[role="button"] {
        background-color: #800080 !important;
        color: white !important;
        border: 1px solid #fff !important;
        border-radius: 5px;
    }
    .section-header {
        background-color: #011F5B;
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .main-title {
        font-size: 50px !important;
        font-weight: 1000 !important;
        color:#002244!important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="main-title"><u>WEEKLY SLEEP ANALYSER ᶻ 𝗓 ᶻ</u></div>', unsafe_allow_html=True)

# Intro
st.markdown('<p class="blue-text">Analyze your weekly sleep pattern and get personalized suggestions.</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Upload CSV
st.markdown('<p style="color:#132257; font-size:18px; font-weight:bold;">Upload 7-day sleep data CSV</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])
st.markdown("<br>", unsafe_allow_html=True)

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("fake_weekly_input.csv")

if df.shape[0] != 7:
    st.error("Please provide exactly 7 days of sleep data.")
    st.stop()

# SECTION 1 - Sleep Duration
st.markdown('<div class="section-header">Sleep Duration Over the Week</div>', unsafe_allow_html=True)

fig_line = px.line(
    df,
    x="date",
    y="total_sleep_hrs",
    markers=True,
    title="Total Sleep Hours per Day",
    labels={"date": "Date", "total_sleep_hrs": "Total Sleep Hours"},
    line_shape="linear",
)
fig_line.update_traces(line=dict(color="#6FA5FC"), marker=dict(color="#6FA5FC"))
fig_line.update_layout(
    plot_bgcolor="#431C76",
    paper_bgcolor="#431C76",
    font=dict(color="#F3CCFF"),
    xaxis=dict(title_font=dict(color="#F3CCFF"), tickfont=dict(color="#F3CCFF"), gridcolor="#663B94"),
    yaxis=dict(title_font=dict(color="#F3CCFF"), tickfont=dict(color="#F3CCFF"), gridcolor="#663B94"),
    legend=dict(font=dict(color="#F3CCFF")),
    title_font=dict(color="#F3CCFF"),
    margin=dict(t=50, b=40, l=40, r=40),
)
st.plotly_chart(fig_line, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# SECTION 2 - Sleep Breakdown
st.markdown('<div class="section-header">Sleep Stage Breakdown (Last Day)</div>', unsafe_allow_html=True)

last_day = df.iloc[-1]
labels = ["Light Sleep", "Deep Sleep", "REM Sleep", "Awake"]
values = [
    last_day["light_sleep_hrs"],
    last_day["deep_sleep_hrs"],
    last_day["rem_sleep_hrs"],
    last_day["awake_hrs"],
]

fig_pie = px.pie(
    values=values,
    names=labels,
    title=f"Sleep Stages on {last_day['date']}",
    color_discrete_sequence=["#DDA0DD", "#6495ED", "#082567", "#7B68EE"],
)
fig_pie.update_layout(
    plot_bgcolor="#431C76",
    paper_bgcolor="#431C76",
    font=dict(color="#F3CCFF"),
    legend=dict(font=dict(color="#F3CCFF")),
    title_font=dict(color="#F3CCFF"),
    margin=dict(t=50, b=40, l=40, r=40),
)
st.plotly_chart(fig_pie, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# SECTION 3 - Sleep Score
st.markdown('<div class="section-header">Sleep Score</div>', unsafe_allow_html=True)

features = [
    "total_sleep_hrs",
    "light_sleep_hrs",
    "deep_sleep_hrs",
    "rem_sleep_hrs",
    "awake_hrs",
    "latency_mins",
    "interruptions",
    "consistency_score",
]
X = df[features]
predictions = model.predict(X)
avg_score = int(predictions.mean())

# Emoji based on score
if avg_score >= 80:
    emoji = "😀"
elif avg_score >= 60:
    emoji = "😐"
else:
    emoji = "😞"

# Score display
st.markdown(f"""
<div style='
    font-size: 20px;
    font-weight: 600;
    color: #080942;
    margin-bottom: 5px;
'>
    Average Sleep Score (out of 100)
</div>
<div style='
    font-size: 40px;
    font-weight: bold;
    color: #33006F;
    margin-top: -10px;
'>
    {avg_score} {emoji}
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# SECTION 4 - Suggestions
st.markdown('<div class="section-header">Personalized Suggestions</div>', unsafe_allow_html=True)

if avg_score >= 90:
    suggestion = "Excellent sleep quality! Keep maintaining your routine."
elif avg_score >= 80:
    suggestion = "You're doing great! Try to reduce screen time before bed."
elif avg_score >= 70:
    suggestion = "Your sleep is decent. Aim for more deep and REM sleep."
elif avg_score >= 60:
    suggestion = "Consider reducing interruptions and sleep latency."
else:
    suggestion = "Poor sleep quality. Improve consistency and sleep hygiene."

st.info(suggestion)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
