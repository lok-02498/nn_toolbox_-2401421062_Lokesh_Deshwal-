import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix

def show():
    st.title("🔗 MLP Classifier")
    st.markdown("""
    A **Multi-Layer Perceptron (MLP)** is a full neural network with:
    - An **input layer** (your features)
    - One or more **hidden layers** 
    - An **output layer** (your predictions)
    """)

    st.divider()

    st.subheader("📂 Choose Dataset")
    dataset_choice = st.selectbox("Dataset", ["Iris (3 classes)", "Breast Cancer (binary)", "Upload CSV"])

    if dataset_choice == "Iris (3 classes)":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        st.info("🌸 Iris dataset: 150 samples, 4 features, 3 flower classes")

    elif dataset_choice == "Breast Cancer (binary)":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        st.info("🏥 Breast Cancer dataset: 569 samples, 30 features, 2 classes (malignant/benign)")

    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.warning("Please upload a CSV file. The last column will be used as the target.")
            return
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        X = df.iloc[:, :-1].select_dtypes(include=[np.number])
        y = df.iloc[:, -1]
        st.info(f"Loaded: {X.shape[0]} rows, {X.shape[1]} features")

    st.divider()
    st.subheader("🛠️ Configure the Network")

    col1, col2, col3 = st.columns(3)
    with col1:
        h1 = st.slider("Hidden Layer 1 neurons", 4, 64, 16)
    with col2:
        h2 = st.slider("Hidden Layer 2 neurons", 0, 64, 8, help="Set to 0 to disable")
    with col3:
        lr = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.5], value=0.01)

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Max Epochs", 50, 500, 200, 50)
    with col2:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)

    activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])

    hidden_layers = (h1,) if h2 == 0 else (h1, h2)

    if st.button("🚀 Train MLP", type="primary"):
        with st.spinner("Training..."):
            X_arr = X.values
            y_arr = y.values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_arr)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_arr, test_size=test_size/100, random_state=42
            )

            clf = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                learning_rate_init=lr,
                max_iter=epochs,
                random_state=42,
                verbose=False
            )
            clf.fit(X_train, y_train)

        st.success("✅ Training complete!")
        st.divider()
        st.subheader("📊 Results")

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        col1, col2, col3 = st.columns(3)
        col1.metric("Train Accuracy", f"{train_acc*100:.1f}%")
        col2.metric("Test Accuracy", f"{test_acc*100:.1f}%")
        col3.metric("Epochs Run", clf.n_iter_)

        # Loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=clf.loss_curve_, mode="lines",
                                      line=dict(color="#3498db", width=2), name="Training Loss"))
        fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss",
                                template="plotly_dark")
        st.plotly_chart(fig_loss, use_container_width=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, clf.predict(X_test))
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           title="Confusion Matrix (Test Set)",
                           labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance via weight magnitude
        st.subheader("🔍 Input Weight Magnitudes (Layer 1)")
        w_magnitudes = np.abs(clf.coefs_[0]).mean(axis=1)
        feat_names = X.columns.tolist()
        fig_feat = px.bar(x=feat_names, y=w_magnitudes,
                          labels={"x": "Feature", "y": "Avg |Weight|"},
                          title="Which features matter most?",
                          color=w_magnitudes, color_continuous_scale="Viridis")
        fig_feat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_feat, use_container_width=True)
