import streamlit as st

st.set_page_config(
    page_title="Neural Network Toolbox",
    layout="wide",
    page_icon="🧠"
)

st.sidebar.title("🧠 NN Toolbox")

page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "⚡ Perceptron",
    "➡️ Forward Propagation",
    "⬅️ Backward Propagation",
    "🔗 MLP Classifier",
    "📈 RNN",
    "🧩 CNN Working Model From Scratch",
    "💬 Sentiment Analysis",
    "📷 Face Detection",
    "🧠 Hopfield OCR",
])

if page == "🏠 Home":
    from src.pages import home
    home.show()

elif page == "⚡ Perceptron":
    from src.pages import perceptron
    perceptron.show()

elif page == "➡️ Forward Propagation":
    from src.pages import forward_prop
    forward_prop.show()

elif page == "⬅️ Backward Propagation":
    from src.pages import backward_prop
    backward_prop.show()

elif page == "🔗 MLP Classifier":
    from src.pages import mlp
    mlp.show()

elif page == "📈 RNN":
    from src.pages import rnn
    rnn.show()

elif page == "🧩 CNN Working Model From Scratch":
    from src.pages import cnn
    cnn.show()

elif page == "💬 Sentiment Analysis":
    from src.pages import sentiment
    sentiment.show()

elif page == "📷 Face Detection":
    from src.pages import face_detection
    face_detection.show()

elif page == "🧠 Hopfield OCR":
    from src.pages import hopfield_ocr
    hopfield_ocr.show()