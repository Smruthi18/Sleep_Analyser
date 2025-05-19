import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page setup
st.set_page_config(page_title="Sleep Analyzer", layout="centered")

# Inject background color and Poppins font via CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, .stApp {
        background-color: #A796E8;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: #1A1A1A;
    }

    .stMarkdown, .stDataFrame, .stTextInput, .stUpload, .stMetric {
        font-family: 'Poppins', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛌 Weekly Sleep Analyzer")
st.markdown("Analyze your weekly sleep pattern and get personalized suggestions.")

# Email input placeholder
email = st.text_input("Enter your email to fetch sleep data (coming soon)", "")

# Upload or fallback to local CSV
uploaded_file = st.file_uploader("Upload 7-day sleep data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("fake_weekly_input.csv")

if df.shape[0] != 7:
    st.error("Please provide exactly 7 days of sleep data.")
    st.stop()

# Line chart: Total Sleep Hours over the week
st.subheader("📈 Sleep Duration Over the Week")
fig_line = px.line(
    df,
    x='date',
    y='total_sleep_hrs',
    markers=True,
    title="Total Sleep Hours per Day",
    labels={"date": "Date", "total_sleep_hrs": "Total Sleep Hours"},
)
st.plotly_chart(fig_line, use_container_width=True)

# Pie chart: Sleep stage breakdown for last day
st.subheader("🧠 Sleep Stage Breakdown (Last Day)")
last_day = df.iloc[-1]
labels = ["Light Sleep", "Deep Sleep", "REM Sleep", "Awake"]
values = [
    last_day["light_sleep_hrs"],
    last_day["deep_sleep_hrs"],
    last_day["rem_sleep_hrs"],
    last_day["awake_hrs"],
]
custom_colors = ['#A2C5F4', '#7FB3D5', '#D6A4E4', '#F4C7C3']
fig_pie = px.pie(
    values=values,
    names=labels,
    title=f"Sleep Stages on {last_day['date']}",
    color_discrete_sequence=custom_colors,
)
fig_pie.update_traces(textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# Load model
model = joblib.load("sleep_model.pkl")

# Sleep score prediction
st.subheader("💤 Sleep Score")
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
st.metric(label="Average Sleep Score (out of 100)", value=avg_score)

# Suggestions based on score
st.subheader("📋 Personalized Suggestions")
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

st.markdown("---")
st.caption("Future feature: Sleep data will be fetched automatically from Google Fit via your email.")
