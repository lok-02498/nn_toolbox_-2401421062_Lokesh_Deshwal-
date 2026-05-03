import streamlit as st
import pickle
import pandas as pd
import re
import streamlit.components.v1 as components


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        saved_objects = pickle.load(f)
    return saved_objects["model"], saved_objects["vectorizer"]


def predict_rating(review, model, vectorizer):
    from preprocessing import preprocess_text
    preprocessed_review = preprocess_text([review])
    review_vectorized   = vectorizer.transform(preprocessed_review)
    prediction  = model.predict(review_vectorized)
    # Try to get probability scores for confidence meter
    try:
        proba = model.predict_proba(review_vectorized)[0]
        confidence = float(max(proba)) * 100
    except Exception:
        confidence = None
    return prediction[0], confidence


# ── simple keyword lists for live text hints ──────────
POSITIVE_WORDS = {
    "amazing","excellent","great","love","perfect","wonderful","fantastic",
    "good","best","awesome","happy","satisfied","recommend","quality",
    "superb","outstanding","brilliant","nice","worthy","value",
}
NEGATIVE_WORDS = {
    "terrible","bad","awful","horrible","worst","poor","hate","disappointing",
    "broken","useless","waste","slow","cheap","wrong","damaged","defective",
    "never","refund","return","problem","issue","complaint","fail","ugly",
}

def live_stats(text: str):
    words  = re.findall(r"\b\w+\b", text.lower())
    pos    = [w for w in words if w in POSITIVE_WORDS]
    neg    = [w for w in words if w in NEGATIVE_WORDS]
    return len(words), len(pos), len(neg), pos, neg


# ─────────────────────────────────────────────────────
# Result HTML card (animated)
# ─────────────────────────────────────────────────────
def result_card_html(is_positive: bool, confidence, review_text: str,
                     pos_words: list, neg_words: list):
    """Full animated result card with sentiment meter, keyword pills, pipeline."""

    label      = "POSITIVE" if is_positive else "NEGATIVE"
    emoji      = "🔥" if is_positive else "💤"
    main_color = "#00e676" if is_positive else "#ff5252"
    bg_color   = "rgba(0,230,118,0.07)" if is_positive else "rgba(255,82,82,0.07)"
    stars      = "⭐⭐⭐⭐⭐" if is_positive else "⭐☆☆☆☆"
    rating_txt = "5 / 5 — Positive" if is_positive else "1–4 / 5 — Negative"
    conf_val   = f"{confidence:.1f}%" if confidence is not None else "N/A"
    conf_num   = confidence if confidence is not None else 75

    # Needle angle: -90° = left (0%), +90° = right (100%)
    needle_deg = -90 + (conf_num / 100) * 180

    # Keyword pills
    def pills(words, color):
        if not words:
            return f'<span style="color:#546e7a;font-size:11px">none detected</span>'
        seen = []
        out  = ""
        for w in words:
            if w not in seen:
                seen.append(w)
                out += f'<span class="kpill" style="border-color:{color};color:{color}">{w}</span>'
        return out

    pos_pills = pills(pos_words, "#00e676")
    neg_pills = pills(neg_words, "#ff5252")

    # Pipeline steps
    pipeline = [
        ("📝", "Raw Text",      "Your review as typed"),
        ("🧹", "Preprocess",    "Lowercase → remove noise → tokenize"),
        ("🔢", "Vectorize",     "TF-IDF: words become numbers"),
        ("🤖", "ML Model",      "Trained classifier scores the vector"),
        ("🎯", "Prediction",    f"{label} ({conf_val} confidence)"),
    ]
    pipeline_html = ""
    for i, (ico, name, desc) in enumerate(pipeline):
        active       = 'pipe-active' if i == len(pipeline)-1 else ''
        active_style = f'border-color:{main_color};background:rgba(0,0,0,.4);box-shadow:0 0 10px {main_color}55' \
                       if i == len(pipeline)-1 else ''
        pipeline_html += f"""<div class="pipe-step {active}" style="{active_style}">
          <div class="pipe-ico">{ico}</div>
          <div class="pipe-name">{name}</div>
          <div class="pipe-desc">{desc}</div>
        </div>"""
        if i < len(pipeline) - 1:
            pipeline_html += f'<div class="pipe-arrow" style="color:{main_color}88">→</div>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:linear-gradient(135deg,#0a0a1a,#12122a,#0a0a1a);
  font-family:'Nunito',sans-serif;color:#e0e0e0;
  padding:16px;min-height:480px;
}}

/* ── VERDICT BANNER ── */
.banner{{
  background:{bg_color};
  border:2px solid {main_color};
  border-radius:18px;padding:18px 22px;
  display:flex;align-items:center;gap:20px;
  animation:popIn .6s cubic-bezier(.36,.07,.19,.97) both;
  margin-bottom:14px;
}}
.big-emoji{{font-size:52px;line-height:1;animation:spin .5s ease both}}
.verdict-text .label{{
  font-family:'Fredoka One',cursive;
  font-size:32px;color:{main_color};
  text-shadow:0 0 20px {main_color}88;
  letter-spacing:1px;
}}
.verdict-text .stars{{font-size:20px;margin:2px 0}}
.verdict-text .rating{{font-size:13px;color:#90a4ae}}

/* ── METER ── */
.meter-wrap{{text-align:center;margin-bottom:14px}}
.meter-title{{font-family:'Fredoka One',cursive;font-size:13px;color:#78909c;
              text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.gauge-svg{{display:block;margin:0 auto}}
.conf-num{{font-family:'Fredoka One',cursive;font-size:26px;color:{main_color};
           text-align:center;margin-top:-8px}}
.conf-sub{{font-size:11px;color:#546e7a;text-align:center}}

/* ── KEYWORD SECTION ── */
.kw-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}}
.kw-card{{
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
  border-radius:12px;padding:10px 12px;
}}
.kw-title{{font-family:'Fredoka One',cursive;font-size:12px;
           text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.kpill{{
  display:inline-block;border:1px solid;border-radius:20px;
  padding:2px 9px;font-size:11px;font-weight:800;
  margin:2px 2px 2px 0;
}}

/* ── PIPELINE ── */
.pipe-title{{font-family:'Fredoka One',cursive;font-size:14px;color:#7986cb;
             text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}}
/* 5 steps + 4 arrows in a strict single row — never wraps */
.pipeline{{
  display:grid;
  grid-template-columns:1fr 20px 1fr 20px 1fr 20px 1fr 20px 1fr;
  align-items:center;gap:4px;
  width:100%;
}}
.pipe-step{{
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);
  border-radius:10px;padding:8px 6px;text-align:center;
  transition:transform .2s;min-width:0;
}}
.pipe-step:hover{{transform:scale(1.05)}}
.pipe-active{{animation:popIn .7s ease both .3s;opacity:0;animation-fill-mode:forwards}}
.pipe-ico{{font-size:16px;margin-bottom:2px}}
.pipe-name{{font-family:'Fredoka One',cursive;font-size:11px;color:#e0e0e0;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.pipe-desc{{font-size:8.5px;color:#546e7a;margin-top:2px;line-height:1.3;
            white-space:normal;word-break:break-word}}
.pipe-arrow{{font-size:14px;color:#37474f;text-align:center;flex-shrink:0}}

@keyframes popIn{{
  0%{{transform:scale(.4);opacity:0}}
  70%{{transform:scale(1.08)}}
  100%{{transform:scale(1);opacity:1}}
}}
@keyframes spin{{
  0%{{transform:rotate(-20deg) scale(.5);opacity:0}}
  100%{{transform:rotate(0deg) scale(1);opacity:1}}
}}
@keyframes needleSpin{{
  0%{{transform:rotate(-90deg)}}
  100%{{transform:rotate({needle_deg}deg)}}
}}
</style></head><body>

<!-- Verdict banner -->
<div class="banner">
  <div class="big-emoji">{emoji}</div>
  <div class="verdict-text">
    <div class="label">{label}</div>
    <div class="stars">{stars}</div>
    <div class="rating">Predicted rating: {rating_txt}</div>
  </div>
</div>

<!-- Confidence meter -->
<div class="meter-wrap">
  <div class="meter-title">🎯 Model Confidence</div>
  <svg class="gauge-svg" width="220" height="120" viewBox="0 0 220 120">
    <defs>
      <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%"   stop-color="#ff5252"/>
        <stop offset="50%"  stop-color="#ffd54f"/>
        <stop offset="100%" stop-color="#00e676"/>
      </linearGradient>
    </defs>
    <!-- track -->
    <path d="M20,110 A90,90 0 0,1 200,110"
          fill="none" stroke="rgba(255,255,255,.1)" stroke-width="14" stroke-linecap="round"/>
    <!-- colored arc -->
    <path d="M20,110 A90,90 0 0,1 200,110"
          fill="none" stroke="url(#arcGrad)" stroke-width="14" stroke-linecap="round"
          opacity="0.85"/>
    <!-- needle -->
    <g transform="translate(110,110)">
      <line id="needle" x1="0" y1="0" x2="0" y2="-75"
            stroke="{main_color}" stroke-width="3" stroke-linecap="round"
            style="transform-origin:0 0;animation:needleSpin .9s cubic-bezier(.4,0,.2,1) both"/>
      <circle cx="0" cy="0" r="7" fill="{main_color}" opacity="0.9"/>
    </g>
    <!-- labels -->
    <text x="14"  y="115" font-size="10" fill="#ff5252"  font-family="Nunito,sans-serif">0%</text>
    <text x="103" y="22"  font-size="10" fill="#ffd54f"  font-family="Nunito,sans-serif" text-anchor="middle">50%</text>
    <text x="194" y="115" font-size="10" fill="#00e676"  font-family="Nunito,sans-serif" text-anchor="end">100%</text>
  </svg>
  <div class="conf-num">{conf_val}</div>
  <div class="conf-sub">confidence in this prediction</div>
</div>

<!-- Keyword pills -->
<div class="kw-grid">
  <div class="kw-card">
    <div class="kw-title" style="color:#00e676">✅ Positive signals found</div>
    {pos_pills}
  </div>
  <div class="kw-card">
    <div class="kw-title" style="color:#ff5252">❌ Negative signals found</div>
    {neg_pills}
  </div>
</div>

<!-- ML pipeline -->
<div class="pipe-title">🔬 How the model reached this conclusion</div>
<div class="pipeline">{pipeline_html}</div>

</body></html>"""


# ─────────────────────────────────────────────────────
# Live text analysis HTML (updates as you type)
# ─────────────────────────────────────────────────────
def live_stats_html(word_count, pos_count, neg_count):
    score    = pos_count - neg_count
    if word_count == 0:
        bar_pct = 50
    else:
        # map score to 0-100 range (clamped)
        bar_pct = max(5, min(95, 50 + score * 8))

    bar_color = "#00e676" if score > 0 else "#ff5252" if score < 0 else "#ffd54f"
    hint = ("Sounds positive! 😊" if score > 0
            else "Sounds negative 😟" if score < 0
            else "Neutral so far 😐")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Nunito',sans-serif;color:#e0e0e0;padding:6px 0}}
.row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.chip{{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.13);
  border-radius:20px;padding:4px 12px;font-size:12px;font-weight:800;
  display:flex;gap:6px;align-items:center;
}}
.chip .num{{font-family:'Fredoka One',cursive;font-size:15px}}
.bar-wrap{{flex:1;min-width:160px;background:rgba(255,255,255,.07);
           border-radius:20px;height:10px;overflow:hidden}}
.bar{{height:100%;border-radius:20px;background:{bar_color};
      width:{bar_pct}%;transition:width .4s ease;
      box-shadow:0 0 8px {bar_color}88}}
.hint{{font-family:'Fredoka One',cursive;font-size:13px;color:{bar_color}}}
</style></head>
<body>
<div class="row">
  <div class="chip"><span>📝</span><span class="num" style="color:#42a5f5">{word_count}</span><span>words</span></div>
  <div class="chip"><span>✅</span><span class="num" style="color:#00e676">{pos_count}</span><span>positive</span></div>
  <div class="chip"><span>❌</span><span class="num" style="color:#ff5252">{neg_count}</span><span>negative</span></div>
  <div class="bar-wrap"><div class="bar"></div></div>
  <div class="hint">{hint}</div>
</div>
</body></html>"""


# ─────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────
def show():
    st.title("💬 Sentiment Analysis")
    st.markdown("""
    **Sentiment Analysis** is a classic NLP task — we teach a machine to read text  
    and decide whether it carries a **positive** or **negative** feeling. 🧠

    Under the hood: your review is *cleaned → vectorized → scored* by a trained ML classifier.
    """)

    # ── How it works (beginner explainer) ──────────────────────────
    with st.expander("📖 How does Sentiment Analysis work? (click to learn)", expanded=False):
        st.markdown("""
| Step | What happens | Example |
|------|-------------|---------|
| **1. Raw text** | You type a review | *"This product is amazing!"* |
| **2. Preprocessing** | Lowercase, remove punctuation, strip stopwords | *"product amazing"* |
| **3. TF-IDF Vectorization** | Each word becomes a number based on how important it is | `[0, 0.83, 0, 0.54, ...]` |
| **4. ML Classifier** | Model (e.g. Logistic Regression / Naive Bayes) scores the vector | `P(positive) = 0.92` |
| **5. Decision** | If score ≥ 0.5 → Positive, else → Negative | **✅ Positive** |

> 💡 **TF-IDF** stands for *Term Frequency × Inverse Document Frequency*.  
> Words that appear a lot in one review but rarely in others get a higher score — they're more *informative*.
        """)

    st.divider()

    # ── Load model ──────────────────────────────────────────────────
    try:
        model, vectorizer = load_model()
        st.success("✅ Model loaded successfully!", icon="🤖")
    except FileNotFoundError:
        st.error("❌ `sentiment_model.pkl` not found. Place it in the same folder as `app.py`.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.divider()

    # ── Text input ──────────────────────────────────────────────────
    st.subheader("✍️ Write Your Review")
    st.caption("The analysis updates live as you type — watch the sentiment bar change!")

    review_input = st.text_area(
        "Your review:",
        height=130,
        placeholder="Example: This product is absolutely amazing and totally worth the price!",
        label_visibility="collapsed",
    )

    # Live stats bar (updates every keystroke via Streamlit re-run)
    wc, pos_c, neg_c, pos_ws, neg_ws = live_stats(review_input)
    components.html(live_stats_html(wc, pos_c, neg_c), height=52, scrolling=False)

    # ── Analyse button ──────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        analyse = st.button("🔍 Analyse Sentiment", use_container_width=True, type="primary")
    with col2:
        clear = st.button("🗑️ Clear", use_container_width=True)

    if clear:
        st.rerun()

    if analyse:
        if not review_input.strip():
            st.warning("⚠️ Please enter a review first.")
        else:
            with st.spinner("Running the ML pipeline..."):
                prediction, confidence = predict_rating(review_input, model, vectorizer)

            is_pos = (prediction == 1)

            st.divider()
            st.subheader("📊 Analysis Result")

            # Animated result card
            html = result_card_html(is_pos, confidence, review_input, pos_ws, neg_ws)
            components.html(html, height=660, scrolling=False)

            if is_pos:
                st.balloons()
            else:
                st.snow()

            # ── What the model "saw" ────────────────────────────────
            st.divider()
            st.subheader("🔬 What the Model Saw")

            with st.expander("🧹 Preprocessed Text", expanded=False):
                try:
                    from preprocessing import preprocess_text
                    cleaned = preprocess_text([review_input])
                    st.code(cleaned[0] if cleaned else "(empty after preprocessing)")
                except Exception as e:
                    st.warning(f"Could not show preprocessed text: {e}")

            with st.expander("🔢 Top TF-IDF Features (most important words)", expanded=False):
                try:
                    from preprocessing import preprocess_text
                    cleaned = preprocess_text([review_input])
                    vec     = vectorizer.transform(cleaned)
                    feature_names = vectorizer.get_feature_names_out()
                    scores  = vec.toarray()[0]
                    top_idx = scores.argsort()[::-1][:15]
                    top_words  = [feature_names[i] for i in top_idx if scores[i] > 0]
                    top_scores = [scores[i]        for i in top_idx if scores[i] > 0]

                    if top_words:
                        import plotly.graph_objects as go
                        colors = ["#00e676" if w in POSITIVE_WORDS
                                  else "#ff5252" if w in NEGATIVE_WORDS
                                  else "#42a5f5"
                                  for w in top_words]
                        fig = go.Figure(go.Bar(
                            x=top_scores, y=top_words,
                            orientation="h",
                            marker=dict(color=colors, opacity=0.85),
                        ))
                        fig.update_layout(
                            title="TF-IDF Scores (higher = more important to the model)",
                            xaxis_title="TF-IDF score",
                            template="plotly_dark",
                            height=max(200, len(top_words) * 30 + 80),
                            margin=dict(l=10, r=10, t=40, b=20),
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("🟢 Green = positive keyword · 🔴 Red = negative keyword · 🔵 Blue = neutral")
                    else:
                        st.info("No strong TF-IDF features found (review may be very short or all stopwords).")
                except Exception as e:
                    st.warning(f"Could not compute TF-IDF breakdown: {e}")

    # ── Dataset viewer ──────────────────────────────────────────────
    st.divider()
    with st.expander("📊 View Training Dataset (Flipkart Reviews)", expanded=False):
        try:
            df = pd.read_excel("flipkardata.xlsx")
            st.markdown(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")

            # Quick summary stats
            c1, c2, c3 = st.columns(3)
            if "Rating" in df.columns or "rating" in df.columns:
                col = "Rating" if "Rating" in df.columns else "rating"
                pos_pct = (df[col] >= 5).mean() * 100
                c1.metric("⭐ Positive samples", f"{pos_pct:.1f}%")
                c2.metric("⭐ Negative samples", f"{100-pos_pct:.1f}%")
            c3.metric("📝 Total reviews", f"{len(df):,}")

            st.dataframe(df.head(10), use_container_width=True)
        except Exception:
            st.warning("⚠️ `flipkardata.xlsx` not found or couldn't be loaded. This is optional.")

    # ── Try it yourself examples ────────────────────────────────────
    st.divider()
    st.subheader("💡 Try these sample reviews")
    examples = [
        ("✅ Positive", "This product is absolutely fantastic! Great quality and fast delivery. Highly recommend to everyone."),
        ("❌ Negative", "Terrible product. Stopped working after 2 days. Complete waste of money. Never buying again."),
        ("😐 Mixed",    "The product is okay. Delivery was fast but the quality could be better for the price."),
    ]
    cols = st.columns(3)
    for col, (label, text) in zip(cols, examples):
        with col:
            st.markdown(f"**{label}**")
            st.caption(text[:80] + "…")
            if st.button(f"Use this ↑", key=label, use_container_width=True):
                st.session_state["_sample_review"] = text
                st.info(f"Copy this into the review box above:\n\n_{text}_")

    st.divider()
    st.caption("Integrated into Neural Network Toolbox · Sentiment model: TF-IDF + ML Classifier")