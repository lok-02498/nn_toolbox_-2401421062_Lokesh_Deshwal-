import streamlit as st
import math
import random
import re


# ---------------- BASIC HELPERS ----------------

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def tanh(x):
    return math.tanh(x)


def softmax(scores):
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]


# ---------------- MAIN PAGE ----------------

def show():
    st.title("🔁 RNN From Scratch")
    st.markdown("### Easy Working Model for Beginners")

    st.markdown("""
    <style>
    .concept-box {
        padding: 18px;
        border-radius: 15px;
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
        border: 1px solid #c7d2fe;
        margin-bottom: 16px;
    }
    .step-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #f8fafc;
        border-left: 6px solid #6366f1;
        margin-bottom: 12px;
    }
    .memory-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #ecfeff;
        border: 1px solid #67e8f9;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-box">
    <h4>🧠 Simple Idea</h4>
    <b>RNN = Current Word + Previous Memory → New Memory</b><br><br>
    RNN reads a sentence one word at a time and keeps memory of previous words.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------------- CONCEPT ----------------

    st.subheader("1️⃣ What is an RNN?")

    st.markdown("""
    An **RNN (Recurrent Neural Network)** is used when the order of data matters.

    Examples:
    - Text prediction
    - Speech recognition
    - Time-series prediction
    - Sentiment analysis

    Example sentence:

    ```text
    recurrent → neural → networks
    ```

    The RNN reads each word one by one and updates its **memory**.
    """)

    st.subheader("2️⃣ Core Formula")

    st.latex(r"h_t = \tanh(x_t + W_h \times h_{t-1} + b)")

    st.markdown("""
    Meaning:

    - **xₜ** = current word value
    - **hₜ₋₁** = previous memory
    - **hₜ** = new memory
    - **Wₕ** = memory importance
    - **b** = bias
    - **tanh** = keeps value between -1 and 1
    """)

    st.info("Beginner meaning: New memory depends on current word + old memory.")

    st.divider()

    # ---------------- DATA ----------------

    st.subheader("3️⃣ Small Training Vocabulary")

    vocab = [
        "recurrent",
        "neural",
        "networks",
        "process",
        "sequences",
        "hidden",
        "state",
        "memory",
        "language",
        "models",
        "predict",
        "words",
        "learn",
        "patterns"
    ]

    st.write(vocab)

    random.seed(7)

    word_values = {
        word: random.uniform(-1, 1)
        for word in vocab
    }

    output_weights = {
        word: random.uniform(-1, 1)
        for word in vocab
    }

    with st.expander("🔢 See word numbers"):
        for word in vocab:
            st.write(f"`{word}` → `{word_values[word]:.3f}`")

    st.divider()

    # ---------------- INPUT ----------------

    st.subheader("4️⃣ Try the RNN")

    sentence = st.text_input(
        "Enter sentence using vocabulary words:",
        "recurrent neural networks"
    )

    actual_next_word = st.selectbox(
        "Choose actual next word for learning:",
        vocab,
        index=vocab.index("process")
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        memory_weight = st.slider("Memory Weight Wₕ", 0.0, 1.0, 0.6, 0.1)

    with col2:
        bias = st.slider("Bias b", -1.0, 1.0, 0.1, 0.1)

    with col3:
        learning_rate = st.slider("Learning Rate", 0.01, 0.50, 0.10, 0.01)

    words = tokenize(sentence)

    if not words:
        st.warning("Please enter a sentence.")
        return

    st.divider()

    # ---------------- FORWARD PASS ----------------

    st.subheader("5️⃣ Forward Pass: Reading Word by Word")

    st.markdown("""
    In forward pass, the RNN reads each word and updates memory.
    """)

    hidden_state = 0.0
    steps = []

    for word in words:
        if word not in word_values:
            st.warning(f"`{word}` is not in vocabulary, so it is skipped.")
            continue

        previous_memory = hidden_state
        current_word_value = word_values[word]

        raw_value = current_word_value + memory_weight * previous_memory + bias
        hidden_state = tanh(raw_value)

        steps.append({
            "word": word,
            "previous_memory": previous_memory,
            "word_value": current_word_value,
            "raw_value": raw_value,
            "new_memory": hidden_state
        })

    if not steps:
        st.error("No valid words found. Please use words from vocabulary.")
        return

    for i, step in enumerate(steps, start=1):
        st.markdown(f"""
        <div class="step-box">
        <h4>Time Step {i}: Word = {step["word"]}</h4>
        Previous Memory = <b>{step["previous_memory"]:.3f}</b><br>
        Current Word Value = <b>{step["word_value"]:.3f}</b><br>
        New Memory = <b>{step["new_memory"]:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.code(f"""
h_t = tanh(x_t + W_h × h_previous + b)

h_t = tanh({step["word_value"]:.3f} + {memory_weight:.3f} × {step["previous_memory"]:.3f} + {bias:.3f})
h_t = tanh({step["raw_value"]:.3f})
h_t = {step["new_memory"]:.3f}
""")

    st.markdown(f"""
    <div class="memory-box">
    <h4>Final Memory</h4>
    After reading the full sentence, final hidden state is:
    <b>{hidden_state:.3f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------------- PREDICTION ----------------

    st.subheader("6️⃣ Prediction: Guess the Next Word")

    st.markdown("""
    Now the RNN uses final memory to predict the next word.
    """)

    st.latex(r"score = final\ memory \times output\ weight")

    scores = []

    for word in vocab:
        score = hidden_state * output_weights[word]
        scores.append(score)

    probabilities = softmax(scores)

    predictions = list(zip(vocab, probabilities))
    predictions.sort(key=lambda x: x[1], reverse=True)

    predicted_word = predictions[0][0]

    col1, col2 = st.columns(2)

    col1.metric("Predicted Word", predicted_word)
    col2.metric("Actual Word", actual_next_word)

    st.markdown("### 📊 Top Predictions")

    for word, probability in predictions[:5]:
        st.write(f"**{word}** → {probability:.3f}")
        st.progress(probability)

    st.divider()

    # ---------------- LOSS ----------------

    st.subheader("7️⃣ Loss: How Wrong Was the Model?")

    st.markdown("""
    Loss tells us how wrong the prediction is.

    If actual word gets high probability, loss is small.  
    If actual word gets low probability, loss is large.
    """)

    actual_probability = dict(predictions)[actual_next_word]
    loss = -math.log(actual_probability + 1e-9)

    st.latex(r"loss = -\log(P(actual\ word))")

    col1, col2 = st.columns(2)

    col1.metric("Probability of Actual Word", f"{actual_probability:.4f}")
    col2.metric("Loss", f"{loss:.4f}")

    st.divider()

    # ---------------- BACKWARD PASS ----------------

    st.subheader("8️⃣ Backward Pass: Learning From Mistake")

    st.markdown("""
    Backward pass means:

    ```text
    prediction error → calculate gradient → update weights
    ```

    Simple error formula:
    """)

    st.latex(r"error = predicted\ probability - actual\ value")

    gradients = {}

    for word, probability in predictions:
        actual_value = 1 if word == actual_next_word else 0
        error = probability - actual_value
        gradient = error * hidden_state
        gradients[word] = gradient

    with st.expander("📉 See gradients"):
        for word in vocab:
            st.write(f"`{word}` gradient = `{gradients[word]:.4f}`")

    st.divider()

    # ---------------- WEIGHT UPDATE ----------------

    st.subheader("9️⃣ Weight Update")

    st.markdown("""
    The model updates output weights using this formula:
    """)

    st.latex(r"new\ weight = old\ weight - learning\ rate \times gradient")

    updated_output_weights = {}

    for word in vocab:
        old_weight = output_weights[word]
        new_weight = old_weight - learning_rate * gradients[word]
        updated_output_weights[word] = new_weight

    old_weight_actual = output_weights[actual_next_word]
    new_weight_actual = updated_output_weights[actual_next_word]

    col1, col2 = st.columns(2)

    col1.metric("Old Weight for Actual Word", f"{old_weight_actual:.4f}")
    col2.metric("New Weight for Actual Word", f"{new_weight_actual:.4f}")

    st.success("One learning step completed.")

    st.divider()

    # ---------------- AFTER UPDATE ----------------

    st.subheader("🔁 Prediction After Learning")

    new_scores = []

    for word in vocab:
        new_score = hidden_state * updated_output_weights[word]
        new_scores.append(new_score)

    new_probabilities = softmax(new_scores)

    new_predictions = list(zip(vocab, new_probabilities))
    new_predictions.sort(key=lambda x: x[1], reverse=True)

    new_actual_probability = dict(new_predictions)[actual_next_word]
    new_loss = -math.log(new_actual_probability + 1e-9)

    col1, col2, col3 = st.columns(3)

    col1.metric("Old Loss", f"{loss:.4f}")
    col2.metric("New Loss", f"{new_loss:.4f}")
    col3.metric("New Prediction", new_predictions[0][0])

    st.markdown("### Updated Top Predictions")

    for word, probability in new_predictions[:5]:
        st.write(f"**{word}** → {probability:.3f}")
        st.progress(probability)

    st.divider()

    # ---------------- SUMMARY ----------------

    st.subheader(" Final Simple Summary")

    st.markdown("""
    <div class="concept-box">
    <b>RNN working in simple words:</b><br><br>

    1. Take one word<br>
    2. Convert it into a number<br>
    3. Combine it with old memory<br>
    4. Create new memory<br>
    5. Repeat for all words<br>
    6. Use final memory to predict next word<br>
    7. Compare prediction with actual answer<br>
    8. Update weights to improve prediction<br>
    </div>
    """, unsafe_allow_html=True)