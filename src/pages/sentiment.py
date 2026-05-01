import streamlit as st
import pickle
import pandas as pd
from preprocessing import preprocess_text


@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        saved_objects = pickle.load(f)

    return saved_objects["model"], saved_objects["vectorizer"]


def predict_rating(review, model, vectorizer):
    preprocessed_review = preprocess_text([review])
    review_vectorized = vectorizer.transform(preprocessed_review)
    prediction = model.predict(review_vectorized)
    return prediction[0]


def show():
    st.title("💬 Sentiment Analysis")
    st.markdown("### Product Review Sentiment Predictor")

    st.info("Enter a product review and the model will predict whether it is positive or negative.")

    try:
        model, vectorizer = load_model()
    except FileNotFoundError:
        st.error("sentiment_model.pkl not found. Place it in the same folder as app.py.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.divider()

    st.subheader("✍️ Enter Review")

    review_input = st.text_area(
        "Write your review here:",
        height=150,
        placeholder="Example: This product is amazing and worth the price!"
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if review_input.strip():
            with st.spinner("Analyzing sentiment..."):
                prediction = predict_rating(review_input, model, vectorizer)

            st.divider()

            if prediction == 1:
                st.success("✅ Positive Review")
                st.markdown("⭐ Rating: **5 or above**")
                st.balloons()
            else:
                st.error("❌ Negative Review")
                st.markdown("⭐ Rating: **Below 5**")
        else:
            st.warning("⚠️ Please enter a review first.")

    st.divider()

    with st.expander("📊 View Training Dataset"):
        try:
            df = pd.read_excel("flipkardata.xlsx")
            st.dataframe(df.head(), use_container_width=True)
            st.write("Dataset Shape:", df.shape)
        except Exception:
            st.warning("flipkardata.xlsx not found or could not be loaded. This is optional.")

    st.caption("Integrated into Neural Network Toolbox")