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

st.title("🧠 Emotion Analyzer")

# Input
text = st.text_area("Enter your text")

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

        # Confidence + probabilities
        probs = model.predict_proba(vec)[0]
        confidence = max(probs)

        # Colored output
        st.markdown(
            f"<h2 style='color:{color}'>Emotion: {prediction} {emoji}</h2>",
            unsafe_allow_html=True
        )

        st.info(f"Confidence: {confidence:.2f}")

        # Bar chart
        prob_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probs
        })

        st.subheader("📊 Emotion Probabilities")
        st.bar_chart(prob_df.set_index("Emotion"))
