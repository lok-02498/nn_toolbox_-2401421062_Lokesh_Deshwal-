import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import re
import time

# ─── Built-in training corpus (no downloads needed) ───────────────────────────
CORPUS = """
The quick brown fox jumps over the lazy dog near the river bank.
Neural networks learn patterns from data by adjusting weights and biases.
Deep learning models use multiple layers to extract complex features.
The brain has billions of neurons connected by synapses that transmit signals.
Machine learning helps computers learn without being explicitly programmed.
Natural language processing enables computers to understand human text.
Recurrent neural networks process sequences by maintaining a hidden state.
The hidden state carries memory of previous words in the sentence.
Training a model requires data examples and a loss function to minimize.
Gradient descent updates weights to reduce prediction error over time.
The quick fox ran through the forest and jumped over the stream.
Language models predict the next word given a sequence of previous words.
Artificial intelligence is transforming how we interact with computers today.
Words are converted to numbers called embeddings before entering the network.
The model learns to associate patterns in sequences with likely next words.
Long short term memory networks solve the vanishing gradient problem well.
Recurrent networks unfold through time during the backpropagation process.
Each neuron applies an activation function to produce its output signal.
The softmax function converts raw scores into a probability distribution.
We pick the word with the highest probability as our next word prediction.
Science and technology advance rapidly with the help of machine learning.
Data is the fuel that powers modern artificial intelligence systems today.
The neural network trains on many examples to learn useful representations.
Words in a sentence depend on previous context for their correct meaning.
A vocabulary maps every unique word to an integer index for processing.
""".strip()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

def build_vocab(tokens):
    words = sorted(set(tokens))
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

def make_sequences(tokens, w2i, seq_len=4):
    xs, ys = [], []
    for i in range(len(tokens) - seq_len):
        xs.append([w2i[t] for t in tokens[i:i+seq_len]])
        ys.append(w2i[tokens[i+seq_len]])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

# ─── RNN Model ────────────────────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn   = nn.RNN(embed_dim, hidden_size, num_layers,
                            batch_first=True, nonlinearity='tanh')
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)           # (B, T, embed_dim)
        out, h = self.rnn(x, h)     # (B, T, hidden)
        out = self.fc(out[:, -1])   # last time-step → (B, vocab)
        return out, h

# ─── Prediction helper ────────────────────────────────────────────────────────
def predict_next(model, text, w2i, i2w, seq_len, top_k=5):
    tokens = tokenize(text)
    known  = [t for t in tokens if t in w2i]
    if len(known) < seq_len:
        return None, f"Need at least {seq_len} known words. Found: {known}"
    ctx = [w2i[t] for t in known[-seq_len:]]
    x   = torch.tensor([ctx])
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits[0], dim=0).numpy()
    top_idx = probs.argsort()[::-1][:top_k]
    return [(i2w[i], float(probs[i])) for i in top_idx], None

# ─── Main page ────────────────────────────────────────────────────────────────
def show():
    st.title("🔁 RNN — Next Word Predictor")

    # ── About RNN (collapsible) ─────────────────────────────────────────────
    with st.expander("📖 How does an RNN work? (click to expand)", expanded=False):
        st.markdown("""
        ### What is an RNN?

        A **Recurrent Neural Network (RNN)** is a neural network designed for **sequences**.
        Unlike a regular network, it has a **hidden state** — a kind of short-term memory
        that carries information from one step to the next.

        ```
        Input words:  [ "neural"  →  "networks"  →  "learn" ]
                           ↓              ↓              ↓
        Hidden state:  h₀ ──→ h₁ ──────→ h₂ ──────→ h₃
                                                       ↓
                                                  Predict next word
        ```

        ### At each time step the RNN does:
        ```
        h_t = tanh( W_x · x_t  +  W_h · h_{t-1}  +  b )
        ```
        - `x_t` = current word (as an embedding vector)
        - `h_{t-1}` = hidden state from the previous step (memory!)
        - `h_t` = new hidden state passed to the next step

        ### Final prediction:
        ```
        output = softmax( W_out · h_last )
        ```
        The **softmax** turns raw scores into probabilities — the word with
        the highest probability is predicted as the next word.

        ### How it learns:
        1. **Forward pass** — run the sequence through the RNN, get prediction
        2. **Calculate loss** — how wrong was the prediction? (Cross-Entropy)
        3. **Backpropagation Through Time (BPTT)** — propagate error backwards through each time step
        4. **Update weights** — gradient descent makes the next prediction a little better

        ### This app trains the RNN right in your browser on a small text corpus — no downloads!
        """)

    st.divider()

    # ── Hyperparameters ─────────────────────────────────────────────────────
    st.subheader("⚙️ Model Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        embed_dim   = st.select_slider("Embedding Size",  [8, 16, 32, 64], value=32)
    with col2:
        hidden_size = st.select_slider("Hidden Size",     [16, 32, 64, 128], value=64)
    with col3:
        num_layers  = st.slider("RNN Layers", 1, 3, 1)
    with col4:
        epochs      = st.slider("Training Epochs", 50, 500, 200, 50)

    seq_len = 4   # context window fixed for simplicity

    # ── Training ─────────────────────────────────────────────────────────────
    train_key = f"{embed_dim}_{hidden_size}_{num_layers}_{epochs}"

    if st.button("🚀 Train RNN Model", type="primary"):
        tokens   = tokenize(CORPUS)
        w2i, i2w = build_vocab(tokens)
        X, Y     = make_sequences(tokens, w2i, seq_len)
        vocab_size = len(w2i)

        model   = SimpleRNN(vocab_size, embed_dim, hidden_size, num_layers)
        opt     = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.CrossEntropyLoss()

        losses = []
        progress = st.progress(0, text="Training…")
        loss_chart_placeholder = st.empty()

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            logits, _ = model(X)
            loss = loss_fn(logits, Y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if (ep + 1) % max(1, epochs // 20) == 0:
                pct = (ep + 1) / epochs
                progress.progress(pct, text=f"Epoch {ep+1}/{epochs} — Loss: {loss.item():.4f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=losses, mode="lines",
                                         line=dict(color="#e74c3c", width=2), name="Loss"))
                fig.update_layout(title="Training Loss", xaxis_title="Epoch",
                                  yaxis_title="Cross-Entropy Loss", template="plotly_dark",
                                  height=250, margin=dict(t=40, b=30))
                loss_chart_placeholder.plotly_chart(fig, use_container_width=True)

        progress.empty()
        st.success(f"✅ Trained! Final loss: {losses[-1]:.4f} | Vocab size: {vocab_size} words")

        st.session_state["rnn_model"]  = model
        st.session_state["rnn_w2i"]    = w2i
        st.session_state["rnn_i2w"]    = i2w
        st.session_state["rnn_losses"] = losses
        st.session_state["rnn_key"]    = train_key

    # ── Prediction UI ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Predict the Next Word")

    if "rnn_model" not in st.session_state:
        st.info("👆 Train the model first, then type a sentence to predict the next word.")
        return

    model  = st.session_state["rnn_model"]
    w2i    = st.session_state["rnn_w2i"]
    i2w    = st.session_state["rnn_i2w"]
    losses = st.session_state["rnn_losses"]

    # Quick-start examples
    st.markdown("**Try an example:**")
    examples = [
        "neural networks learn patterns",
        "the quick brown fox",
        "recurrent neural networks process",
        "gradient descent updates weights",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key=ex):
            st.session_state["rnn_input"] = ex

    user_input = st.text_input(
        "Enter a sentence (at least 4 words):",
        value=st.session_state.get("rnn_input", "neural networks learn patterns"),
        key="rnn_text_input"
    )

    top_k = st.slider("Show top-K predictions", 3, 10, 5)

    if user_input:
        results, err = predict_next(model, user_input, w2i, i2w, seq_len, top_k)

        if err:
            st.warning(err)
        else:
            st.markdown(f"#### Input: *\"{user_input}\"*")
            st.markdown(f"#### Predicted next word: **`{results[0][0]}`**")

            # Bar chart of probabilities
            words  = [r[0] for r in results]
            probs  = [r[1] for r in results]
            colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(words))]

            fig = go.Figure(go.Bar(
                x=probs, y=words, orientation="h",
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside"
            ))
            fig.update_layout(
                title="Top Predictions (probability)",
                xaxis_title="Probability",
                yaxis=dict(autorange="reversed"),
                template="plotly_dark",
                height=300,
                xaxis_range=[0, max(probs) * 1.3]
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sentence completion
            st.markdown("**Sentence completions (greedy, 5 words ahead):**")
            sentence = user_input.strip()
            tokens_so_far = tokenize(sentence)
            completed = list(tokens_so_far)

            for _ in range(5):
                res, e = predict_next(model, " ".join(completed), w2i, i2w, seq_len, 1)
                if e or not res:
                    break
                completed.append(res[0][0])

            added = completed[len(tokens_so_far):]
            original_display = " ".join(tokens_so_far)
            added_display = " ".join(added)
            st.markdown(
                f'<p style="font-size:18px">'
                f'{original_display} <span style="color:#2ecc71; font-weight:bold;">{added_display}</span>'
                f'</p>',
                unsafe_allow_html=True
            )
            st.caption("🟢 Green = RNN-generated continuation")

    # ── Architecture diagram ────────────────────────────────────────────────
    st.divider()
    st.subheader("🏗️ Model Architecture")
    vocab_size = len(w2i)
    total_params = sum(p.numel() for p in model.parameters())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vocab Size",    vocab_size)
    col2.metric("Embedding Dim", embed_dim)
    col3.metric("Hidden Size",   hidden_size)
    col4.metric("Total Params",  f"{total_params:,}")

    st.code(f"""
SimpleRNN(
  (embed): Embedding({vocab_size}, {embed_dim})          # word → vector
  (rnn):   RNN({embed_dim}, {hidden_size},               # sequence processing
               num_layers={num_layers}, batch_first=True)
  (fc):    Linear({hidden_size}, {vocab_size})           # hidden → word scores
)
    """, language="python")
