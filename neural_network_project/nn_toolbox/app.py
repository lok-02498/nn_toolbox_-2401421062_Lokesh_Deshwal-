import streamlit as st

st.set_page_config(page_title="Neural Network Toolbox", layout="wide", page_icon="🧠")

# Sidebar navigation
st.sidebar.title("🧠 NN Toolbox")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "⚡ Perceptron",
    "➡️ Forward Propagation",
    "⬅️ Backward Propagation",
    "🔗 MLP Classifier",
])

# Route to pages
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
