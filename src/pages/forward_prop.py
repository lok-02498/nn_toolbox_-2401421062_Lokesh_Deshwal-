import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def show():
    st.title("➡️ Forward Propagation")
    st.markdown("""
    **Forward Propagation** is the process of passing input data through each layer of a neural network 
    to get a final prediction.

    Each neuron: `z = (weights × inputs) + bias`, then apply **activation function** → output
    """)

    st.divider()

    st.subheader("🏗️ Network Configuration")

    col1, col2 = st.columns(2)
    with col1:
        activation = st.selectbox("Activation Function", ["Sigmoid", "ReLU"])
    with col2:
        st.markdown("**Network: 2 inputs → 3 hidden neurons → 1 output**")

    st.divider()

    st.subheader("🎛️ Set Your Inputs")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.slider("Input x1", 0.0, 1.0, 0.5, 0.01)
    with col2:
        x2 = st.slider("Input x2", 0.0, 1.0, 0.8, 0.01)

    # Fixed weights for simplicity
    W1 = np.array([[0.2, 0.8], [0.4, 0.6], [-0.5, 0.9]])
    b1 = np.array([0.1, -0.1, 0.2])
    W2 = np.array([[0.3, -0.4, 0.7]])
    b2 = np.array([0.05])

    X = np.array([x1, x2])
    act_fn = sigmoid if activation == "Sigmoid" else relu

    # Layer 1
    z1 = W1 @ X + b1
    a1 = act_fn(z1)

    # Layer 2 (output)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)  # Always sigmoid at output

    st.divider()
    st.subheader("📊 Step-by-Step Calculation")

    st.markdown("#### Layer 1 — Hidden Layer")
    for i in range(3):
        st.markdown(f"""
        **Neuron {i+1}:**  
        `z = {W1[i,0]:.1f}×{x1:.2f} + {W1[i,1]:.1f}×{x2:.2f} + {b1[i]:.1f} = {z1[i]:.4f}`  
        `a = {activation}({z1[i]:.4f}) = {a1[i]:.4f}`
        """)

    st.markdown("#### Layer 2 — Output Layer")
    st.markdown(f"""
    **Output Neuron:**  
    `z = {W2[0,0]:.1f}×{a1[0]:.4f} + {W2[0,1]:.1f}×{a1[1]:.4f} + {W2[0,2]:.1f}×{a1[2]:.4f} + {b2[0]:.2f} = {z2[0]:.4f}`  
    `output = sigmoid({z2[0]:.4f}) = **{a2[0]:.4f}**`
    """)

    st.metric("🎯 Final Prediction", f"{a2[0]:.4f}", help="Value between 0 and 1")

    # Sweep x1 from 0 to 1 for visualization
    st.divider()
    st.subheader("📈 Output vs Input x1 (x2 fixed)")

    x1_vals = np.linspace(0, 1, 100)
    outputs = []
    for v in x1_vals:
        Xv = np.array([v, x2])
        z1v = W1 @ Xv + b1
        a1v = act_fn(z1v)
        z2v = W2 @ a1v + b2
        outputs.append(sigmoid(z2v)[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1_vals, y=outputs, mode="lines", name="Network Output",
                             line=dict(color="#3498db", width=2)))
    fig.add_vline(x=x1, line_dash="dash", line_color="orange", annotation_text=f"x1={x1:.2f}")
    fig.update_layout(title="Forward Pass Output", xaxis_title="x1", yaxis_title="Output",
                      template="plotly_dark", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
