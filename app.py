import streamlit as st
import joblib
import pandas as pd

# Load model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# Emojis
emotion_emoji = {
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😊"
}

# Colors
color_map = {
    "joy": "green",
    "sadness": "blue",
    "anger": "red",
    "fear": "orange",
    "love": "pink",
    "surprise": "purple"
}

# Emotion order
emotions = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 Emotion Analyzer")

# Example buttons
st.subheader("Try Examples 👇")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("😊 Happy"):
        st.session_state.text = "I am so happy today!"
with col2:
    if st.button("😢 Sad"):
        st.session_state.text = "I feel very lonely and sad"
with col3:
    if st.button("😡 Angry"):
        st.session_state.text = "I am so frustrated right now"
with col4:
    if st.button("😨 Fear"):
        st.session_state.text = "This situation is terrifying"
with col5:
    if st.button("😊 Joy"):
        st.session_state.text = "We are having fun."

text = st.text_area("Enter your text", value=st.session_state.get("text", ""))

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Enter something first")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        pred_index = int(pred)
        prediction = emotions[pred_index]

        emoji = emotion_emoji.get(prediction, "🤔")
        color = color_map.get(prediction, "black")

        # ✅ Confidence + probabilities
        probs = model.predict_proba(vec)[0]
        confidence = max(probs)

        # ✅ Colored output
        st.markdown(
            f"<h2 style='color:{color}'>Emotion: {prediction} {emoji}</h2>",
            unsafe_allow_html=True
        )

        st.info(f"Confidence: {confidence:.2f}")

        # ✅ Bar chart
        prob_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probs
        })

        st.subheader("📊 Emotion Probabilities")
        st.bar_chart(prob_df.set_index("Emotion"))

        # ✅ Save to history
        st.session_state.history.append((text, prediction))

# ✅ Show history
st.subheader("🕘 History")

for t, p in st.session_state.history[::-1]:
    st.write(f"{t} → {p}")

# ✅ Clear history button
if st.button("Clear History"):
    st.session_state.history = []
