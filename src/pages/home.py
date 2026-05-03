import streamlit as st

def show():
    st.title("🧠 Neural Network Toolbox")
    st.markdown("### An interactive learning platform for Neural Networks")

    st.markdown("""
    Welcome! This toolbox helps you **understand neural networks step by step** with hands-on demos.

    ---
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📚 What you can learn:")
        st.markdown("""
        - ⚡ **Perceptron** — The building block of all neural networks
        - ➡️ **Forward Propagation** — How data flows through a network
        - ⬅️ **Backward Propagation** — How a network learns from errors
        - 🔗 **MLP Classifier** — A full multi-layer neural network
        """)

    with col2:
        st.markdown("#### 🚀 How to use:")
        st.markdown("""
        1. Use the **sidebar** on the left to pick a topic
        2. Adjust sliders and inputs to experiment
        3. Watch results update in real time
        4. Read explanations alongside each demo
        """)

    st.info("👈 Select a topic from the sidebar to get started!")
