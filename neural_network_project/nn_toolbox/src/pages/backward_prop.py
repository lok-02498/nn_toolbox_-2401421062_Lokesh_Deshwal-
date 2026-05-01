import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def show():
    st.title("⬅️ Backward Propagation")
    st.markdown("""
    **Backpropagation** is how a neural network **learns from its mistakes**.

    After a forward pass, we calculate the error, then propagate it **backwards** through the network
    to update weights using **gradient descent**.
    """)

    st.divider()

    st.subheader("🎛️ Set Up the Problem")
    col1, col2, col3 = st.columns(3)
    with col1:
        x1 = st.slider("Input x1", 0.0, 1.0, 0.5, 0.01)
    with col2:
        x2 = st.slider("Input x2", 0.0, 1.0, 0.3, 0.01)
    with col3:
        target = st.slider("Target Output (y)", 0.0, 1.0, 1.0, 0.01)

    lr = st.slider("Learning Rate (α)", 0.01, 2.0, 0.5, 0.01)

    # Simple 2→2→1 network, fixed init weights
    np.random.seed(42)
    W1 = np.array([[0.15, 0.25], [0.20, 0.30]])
    b1 = np.array([0.35, 0.35])
    W2 = np.array([[0.40, 0.50]])
    b2 = np.array([0.60])

    X = np.array([x1, x2])

    # Forward pass
    z1 = W1 @ X + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    loss = 0.5 * (target - a2[0])**2

    st.divider()
    st.subheader("➡️ Step 1: Forward Pass")

    col1, col2, col3 = st.columns(3)
    col1.metric("Hidden Neuron 1", f"{a1[0]:.4f}")
    col2.metric("Hidden Neuron 2", f"{a1[1]:.4f}")
    col3.metric("Output", f"{a2[0]:.4f}")

    st.metric("📉 Loss (MSE)", f"{loss:.6f}", help="½ × (target - output)²")

    st.divider()
    st.subheader("⬅️ Step 2: Backward Pass (Gradients)")

    # Backprop
    dL_da2 = -(target - a2[0])
    da2_dz2 = sigmoid_deriv(z2[0])
    delta2 = dL_da2 * da2_dz2

    dW2 = delta2 * a1
    db2 = delta2

    da1_dz1 = sigmoid_deriv(z1)
    delta1 = (W2[0] * delta2) * da1_dz1

    dW1 = np.outer(delta1, X)
    db1 = delta1

    st.markdown(f"""
    **Output layer gradient:**  
    `δ_output = (output - target) × sigmoid'(z2) = {dL_da2:.4f} × {da2_dz2:.4f} = {delta2:.4f}`

    **Weight gradients (W2):** `{dW2[0]:.4f}, {dW2[1]:.4f}`  
    **Bias gradient (b2):** `{db2:.4f}`

    **Hidden layer gradient:**  
    `δ_hidden = W2 × δ_output × sigmoid'(z1) = [{delta1[0]:.4f}, {delta1[1]:.4f}]`
    """)

    st.divider()
    st.subheader("🔄 Step 3: Weight Update")

    W1_new = W1 - lr * dW1
    b1_new = b1 - lr * db1
    W2_new = W2 - lr * dW2
    b2_new = b2 - lr * db2

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Update (W2):**")
        st.code(f"[{W2[0,0]:.4f}, {W2[0,1]:.4f}]")
    with col2:
        st.markdown("**After Update (W2):**")
        st.code(f"[{W2_new[0,0]:.4f}, {W2_new[0,1]:.4f}]")

    # Verify: new loss is lower?
    z1n = W1_new @ X + b1_new
    a1n = sigmoid(z1n)
    z2n = W2_new @ a1n + b2_new
    a2n = sigmoid(z2n)
    new_loss = 0.5 * (target - a2n[0])**2

    col1, col2 = st.columns(2)
    col1.metric("Old Loss", f"{loss:.6f}")
    col2.metric("New Loss", f"{new_loss:.6f}", delta=f"{new_loss - loss:.6f}")

    if new_loss < loss:
        st.success("✅ Loss decreased — the network learned!")
    else:
        st.warning("⚠️ Loss increased — try a smaller learning rate.")

    st.divider()
    st.subheader("📉 Loss Over 50 Training Iterations")

    # Simulate training
    W1t, b1t, W2t, b2t = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    losses = []
    for _ in range(50):
        z1t = W1t @ X + b1t
        a1t = sigmoid(z1t)
        z2t = W2t @ a1t + b2t
        a2t = sigmoid(z2t)
        lt = 0.5 * (target - a2t[0])**2
        losses.append(lt)

        d2 = -(target - a2t[0]) * sigmoid_deriv(z2t[0])
        W2t -= lr * d2 * a1t
        b2t -= lr * d2
        d1 = (W2t[0] * d2) * sigmoid_deriv(z1t)
        W1t -= lr * np.outer(d1, X)
        b1t -= lr * d1

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode="lines+markers", name="Loss",
                             line=dict(color="#e74c3c", width=2), marker=dict(size=4)))
    fig.update_layout(title="Loss Decreasing Over Time", xaxis_title="Iteration",
                      yaxis_title="Loss", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
