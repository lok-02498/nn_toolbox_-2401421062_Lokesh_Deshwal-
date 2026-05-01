import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    return 1 if x >= 0.5 else 0

def show():
    st.title("⚡ Perceptron")
    st.markdown("""
    A **Perceptron** is the simplest neural network — it takes inputs, multiplies by weights, 
    adds a bias, and produces an output (0 or 1).

    **Formula:** `output = step(w1*x1 + w2*x2 + bias)`
    """)

    st.divider()

    # Choose gate
    st.subheader("🔧 Logic Gate Demo")
    gate = st.selectbox("Choose a logic gate to simulate:", ["AND", "OR", "NAND", "NOR"])

    gate_data = {
        "AND":  {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [0, 0, 0, 1]},
        "OR":   {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [0, 1, 1, 1]},
        "NAND": {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [1, 1, 1, 0]},
        "NOR":  {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [1, 0, 0, 0]},
    }

    data = gate_data[gate]
    inputs = np.array(data["inputs"])
    targets = np.array(data["targets"])

    st.markdown(f"**Truth table for {gate} gate:**")
    import pandas as pd
    df = pd.DataFrame(inputs, columns=["x1", "x2"])
    df["Expected Output"] = targets
    st.dataframe(df, use_container_width=True)

    st.divider()

    # Manual weights
    st.subheader("🎛️ Tune Weights & Bias Manually")
    col1, col2, col3 = st.columns(3)
    with col1:
        w1 = st.slider("Weight 1 (w1)", -2.0, 2.0, 1.0, 0.1)
    with col2:
        w2 = st.slider("Weight 2 (w2)", -2.0, 2.0, 1.0, 0.1)
    with col3:
        bias = st.slider("Bias", -2.0, 2.0, -1.5, 0.1)

    # Predict with current weights
    predictions = []
    for x in inputs:
        z = w1 * x[0] + w2 * x[1] + bias
        out = step(sigmoid(z))
        predictions.append(out)

    df["Your Prediction"] = predictions
    df["✅ Correct?"] = df["Expected Output"] == df["Your Prediction"]
    accuracy = df["✅ Correct?"].mean() * 100

    st.markdown(f"### Accuracy with your weights: **{accuracy:.0f}%**")
    st.dataframe(df, use_container_width=True)

    if accuracy == 100:
        st.success("🎉 Perfect! Your weights correctly simulate the gate!")
    else:
        st.warning("❌ Not quite — try adjusting the weights and bias!")

    st.divider()

    # Auto-train
    st.subheader("🤖 Auto-Train the Perceptron")
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
    epochs = st.slider("Epochs", 10, 500, 100, 10)

    if st.button("Train Now"):
        w = np.array([0.0, 0.0])
        b = 0.0
        loss_history = []

        for _ in range(epochs):
            total_loss = 0
            for x, y in zip(inputs, targets):
                z = np.dot(w, x) + b
                pred = sigmoid(z)
                error = y - pred
                w += lr * error * x
                b += lr * error
                total_loss += error**2
            loss_history.append(total_loss)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss_history, mode="lines", name="Loss", line=dict(color="#e74c3c")))
        fig.update_layout(title="Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Final predictions
        final_preds = []
        for x in inputs:
            z = np.dot(w, x) + b
            final_preds.append(step(sigmoid(z)))

        df["Trained Prediction"] = final_preds
        acc = (np.array(final_preds) == targets).mean() * 100
        st.markdown(f"### Final Accuracy after training: **{acc:.0f}%**")
        st.dataframe(df[["x1", "x2", "Expected Output", "Trained Prediction"]], use_container_width=True)
        st.success(f"Trained weights: w1={w[0]:.3f}, w2={w[1]:.3f}, bias={b:.3f}")
