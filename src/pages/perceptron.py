import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd


# ✅ True perceptron activation
def step(x):
    return 1 if x >= 0 else 0


def show():
    st.title("⚡ Perceptron")
    st.markdown("""
    A **Perceptron** is the simplest neural network.

    It computes:
    
    `z = w1*x1 + w2*x2 + bias`
    
    Then applies:
    
    `output = step(z)`
    """)

    st.divider()

    # ---------------- LOGIC GATES ----------------
    st.subheader("🔧 Logic Gate Demo")

    gate = st.selectbox(
        "Choose a logic gate to simulate:",
        ["AND", "OR", "NAND", "NOR"]
    )

    gate_data = {
        "AND":  {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [0,0,0,1]},
        "OR":   {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [0,1,1,1]},
        "NAND": {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [1,1,1,0]},
        "NOR":  {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [1,0,0,0]},
    }

    data = gate_data[gate]
    inputs = np.array(data["inputs"])
    targets = np.array(data["targets"])

    df = pd.DataFrame(inputs, columns=["x1", "x2"])
    df["Expected Output"] = targets

    st.dataframe(df, use_container_width=True)

    st.divider()

    # ---------------- MANUAL WEIGHTS ----------------
    st.subheader("🎛️ Tune Weights & Bias Manually")

    col1, col2, col3 = st.columns(3)

    with col1:
        w1 = st.slider("Weight 1 (w1)", -2.0, 2.0, 1.0, 0.1)

    with col2:
        w2 = st.slider("Weight 2 (w2)", -2.0, 2.0, 1.0, 0.1)

    with col3:
        bias = st.slider("Bias", -2.0, 2.0, -1.0, 0.1)

    # Predict manually
    predictions = []
    for x in inputs:
        z = w1 * x[0] + w2 * x[1] + bias
        out = step(z)
        predictions.append(out)

    df["Your Prediction"] = predictions
    df["✅ Correct?"] = df["Expected Output"] == df["Your Prediction"]

    accuracy = df["✅ Correct?"].mean() * 100

    st.markdown(f"### Accuracy with your weights: **{accuracy:.0f}%**")
    st.dataframe(df, use_container_width=True)

    if accuracy == 100:
        st.success("🎉 Perfect! Your weights correctly simulate the gate!")
    else:
        st.warning("❌ Adjust weights to match expected output")

    st.divider()

    # ---------------- AUTO TRAIN ----------------
    st.subheader("🤖 Auto-Train Perceptron")

    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
    epochs = st.slider("Epochs", 10, 500, 100, 10)

    if st.button("Train Now"):
        w = np.zeros(2)
        b = 0.0

        loss_history = []

        for _ in range(epochs):
            total_error = 0

            for x, y in zip(inputs, targets):
                z = np.dot(w, x) + b
                pred = step(z)

                error = y - pred

                # Perceptron update rule
                w += lr * error * x
                b += lr * error

                total_error += abs(error)

            loss_history.append(total_error)

        # Plot loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=loss_history,
            mode="lines",
            name="Error",
            line=dict(color="#e74c3c", width=2)
        ))

        fig.update_layout(
            title="Training Error Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Total Error",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Final predictions
        final_preds = []
        for x in inputs:
            z = np.dot(w, x) + b
            final_preds.append(step(z))

        df["Trained Prediction"] = final_preds

        acc = (np.array(final_preds) == targets).mean() * 100

        st.markdown(f"### Final Accuracy after training: **{acc:.0f}%**")
        st.dataframe(
            df[["x1", "x2", "Expected Output", "Trained Prediction"]],
            use_container_width=True
        )

        st.success(
            f"Final Weights → w1={w[0]:.3f}, w2={w[1]:.3f}, bias={b:.3f}"
        )

    st.divider()

    # ---------------- NOTE ----------------
    st.info("""
    ⚠️ Note:
    - Perceptron works only for **linearly separable problems**
    - It cannot learn XOR
    """)