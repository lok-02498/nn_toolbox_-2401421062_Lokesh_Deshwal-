import streamlit as st
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)


def forward_prop_html(x1, x2, a1, a2_val, z1, z2_val, activation):
    """Animated forward propagation network diagram."""

    # Node activation → color intensity
    def act_color(val, base="52,152,219"):
        intensity = min(1.0, max(0.1, float(val)))
        alpha = 0.25 + intensity * 0.75
        return f"rgba({base},{alpha:.2f})"

    def glow(val, color):
        s = min(1.0, max(0.0, float(val)))
        return f"0 0 {int(12+s*30)}px {color}, 0 0 {int(6+s*15)}px {color}88"

    in_c   = "52,152,219"
    h_c    = "230,126,34"
    out_c  = "46,204,113" if a2_val >= 0.5 else "231,76,60"
    out_hex= "#2ecc71"    if a2_val >= 0.5 else "#e74c3c"

    # hidden node colors based on activation value
    h_colors = [act_color(v, h_c) for v in a1]
    h_glows  = [glow(v, "#e67e22") for v in a1]

    # build JS arrays for wire data
    # wires: input→hidden (6 wires), hidden→output (3 wires)
    W1 = [[0.2, 0.8], [0.4, 0.6], [-0.5, 0.9]]
    W2 = [0.3, -0.4, 0.7]

    def wcolor(w):
        return "#00e676" if w >= 0 else "#ff5252"

    def wthick(w):
        return max(1.5, abs(w) * 4)

    # Encode as JS arrays
    ih_wires = []
    for i in range(3):
        for j in range(2):
            w = W1[i][j]
            src_val = x1 if j == 0 else x2
            ih_wires.append({
                "from": f"in{j}",
                "to": f"h{i}",
                "color": wcolor(w),
                "width": wthick(w),
                "label": f"w={w:+.1f}",
                "signal": float(src_val),
            })

    ho_wires = []
    for i in range(3):
        w = W2[i]
        ho_wires.append({
            "from": f"h{i}",
            "to": "out",
            "color": wcolor(w),
            "width": wthick(w),
            "label": f"w={w:+.1f}",
            "signal": float(a1[i]),
        })

    import json
    ih_js = json.dumps(ih_wires)
    ho_js = json.dumps(ho_wires)

    h1_vals = json.dumps([f"{v:.3f}" for v in a1])
    z1_vals = json.dumps([f"{v:.3f}" for v in z1])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:linear-gradient(135deg,#0a0a1a,#12122a,#0a0a1a);font-family:'Nunito',sans-serif;color:white;padding:14px;min-height:560px}}
.title{{font-family:'Fredoka One',cursive;font-size:20px;text-align:center;color:#ffd54f;margin-bottom:4px}}
.subtitle{{text-align:center;font-size:12px;color:#90a4ae;margin-bottom:10px}}
.canvas{{position:relative;width:100%;height:320px}}
svg.wires{{position:absolute;top:0;left:0;width:100%;height:100%;overflow:visible;pointer-events:none}}
.node{{position:absolute;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:50%;font-weight:800;text-align:center;transition:transform .3s,box-shadow .4s;cursor:default}}
.node:hover{{transform:scale(1.18)!important;z-index:10}}
.node-in{{width:72px;height:72px;font-size:13px;border:2.5px solid #64b5f6}}
.node-h{{width:76px;height:76px;font-size:12px;border:2.5px solid #ffb74d;animation:hpulse 2s ease-in-out infinite}}
.node-h:nth-child(2){{animation-delay:.4s}}
.node-h:nth-child(3){{animation-delay:.8s}}
.node-out{{width:84px;height:84px;font-size:13px;font-weight:800;border:3px solid {out_hex};animation:outpop .7s cubic-bezier(.36,.07,.19,.97) both}}
.node-label{{font-size:9px;font-weight:700;opacity:.8;margin-top:2px;text-transform:uppercase;letter-spacing:.5px}}
.layer-label{{position:absolute;top:4px;font-family:'Fredoka One',cursive;font-size:13px;color:#78909c;text-align:center;width:90px}}
.wire-tag{{position:absolute;background:rgba(5,5,20,.85);border:1px solid rgba(255,255,255,.2);border-radius:6px;padding:1px 5px;font-size:10px;font-weight:800;pointer-events:none;white-space:nowrap}}
.info-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px}}
.info-card{{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:10px 12px;backdrop-filter:blur(6px)}}
.card-title{{font-family:'Fredoka One',cursive;font-size:12px;color:#90a4ae;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}}
.card-val{{font-family:'Fredoka One',cursive;font-size:17px}}
.card-sub{{font-size:10px;color:#78909c;margin-top:2px}}
.output-banner{{background:linear-gradient(90deg,{out_hex}22,{out_hex}44,{out_hex}22);border:1.5px solid {out_hex};border-radius:12px;padding:10px;text-align:center;margin-top:8px;animation:bounceIn .6s ease both}}
.output-banner .big{{font-family:'Fredoka One',cursive;font-size:22px;color:{out_hex}}}
.output-banner .small{{font-size:12px;color:#b0bec5;margin-top:2px}}
@keyframes hpulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.06)}}}}
@keyframes outpop{{0%{{transform:scale(.4);opacity:0}}70%{{transform:scale(1.12)}}100%{{transform:scale(1);opacity:1}}}}
@keyframes bounceIn{{0%{{transform:scale(.3);opacity:0}}60%{{transform:scale(1.08)}}100%{{transform:scale(1);opacity:1}}}}
</style></head><body>

<div class="title">🔄 Forward Propagation — Live!</div>
<div class="subtitle">Watch signals travel from inputs → hidden layer → output in real time</div>

<div class="canvas" id="canvas">
  <svg class="wires" id="wires"></svg>

  <!-- Layer labels -->
  <div class="layer-label" style="left:2%;transform:translateX(0)">Input<br>Layer</div>
  <div class="layer-label" style="left:37%;transform:translateX(0)">Hidden<br>Layer</div>
  <div class="layer-label" style="left:73%;transform:translateX(0)">Output<br>Layer</div>

  <!-- Input nodes -->
  <div class="node node-in" id="in0"
       style="left:4%;top:28%;background:radial-gradient(circle at 35% 35%,rgba({in_c},0.9),rgba({in_c},0.4));box-shadow:{glow(x1, '#3498db')}">
    <div>x1</div><div style="color:#ffd54f;font-size:15px">{x1:.2f}</div>
    <div class="node-label">Input 1</div>
  </div>
  <div class="node node-in" id="in1"
       style="left:4%;top:62%;background:radial-gradient(circle at 35% 35%,rgba({in_c},0.9),rgba({in_c},0.4));box-shadow:{glow(x2, '#3498db')}">
    <div>x2</div><div style="color:#ffd54f;font-size:15px">{x2:.2f}</div>
    <div class="node-label">Input 2</div>
  </div>

  <!-- Hidden nodes -->
  <div class="node node-h" id="h0"
       style="left:36%;top:12%;background:radial-gradient(circle at 35% 35%,{h_colors[0]},{h_colors[0]});box-shadow:{h_glows[0]}">
    <div style="font-size:10px">z={z1[0]:.2f}</div>
    <div style="color:#ffd54f;font-size:14px">{a1[0]:.3f}</div>
    <div class="node-label">H1</div>
  </div>
  <div class="node node-h" id="h1"
       style="left:36%;top:43%;background:radial-gradient(circle at 35% 35%,{h_colors[1]},{h_colors[1]});box-shadow:{h_glows[1]}">
    <div style="font-size:10px">z={z1[1]:.2f}</div>
    <div style="color:#ffd54f;font-size:14px">{a1[1]:.3f}</div>
    <div class="node-label">H2</div>
  </div>
  <div class="node node-h" id="h2"
       style="left:36%;top:74%;background:radial-gradient(circle at 35% 35%,{h_colors[2]},{h_colors[2]});box-shadow:{h_glows[2]}">
    <div style="font-size:10px">z={z1[2]:.2f}</div>
    <div style="color:#ffd54f;font-size:14px">{a1[2]:.3f}</div>
    <div class="node-label">H3</div>
  </div>

  <!-- Output node -->
  <div class="node node-out" id="out"
       style="left:73%;top:40%;background:radial-gradient(circle at 35% 35%,{out_hex}cc,{out_hex}33);box-shadow:{glow(a2_val, out_hex)}">
    <div style="font-size:10px">z={z2_val:.2f}</div>
    <div style="font-size:18px;color:white">{a2_val:.3f}</div>
    <div class="node-label">Output</div>
  </div>
</div>

<!-- Info cards -->
<div class="info-grid">
  <div class="info-card">
    <div class="card-title">⬅️ Inputs</div>
    <div class="card-val" style="color:#42a5f5">x1={x1:.2f} x2={x2:.2f}</div>
    <div class="card-sub">fed into the network</div>
  </div>
  <div class="info-card">
    <div class="card-title">🟠 Hidden ({activation})</div>
    <div class="card-val" style="color:#ffa726">[{a1[0]:.2f}, {a1[1]:.2f}, {a1[2]:.2f}]</div>
    <div class="card-sub">3 neurons activated</div>
  </div>
  <div class="info-card">
    <div class="card-title">🎯 Output (Sigmoid)</div>
    <div class="card-val" style="color:{out_hex}">{a2_val:.4f}</div>
    <div class="card-sub">{"High confidence ✅" if a2_val >= 0.5 else "Low confidence ❌"}</div>
  </div>
</div>

<div class="output-banner">
  <div class="big">{"🔥 Prediction: 1 (Positive!)" if a2_val >= 0.5 else "💤 Prediction: 0 (Negative)"}</div>
  <div class="small">Output = {a2_val:.4f} — {"above" if a2_val >= 0.5 else "below"} the 0.5 decision threshold</div>
</div>

<script>
const ihWires = {ih_js};
const hoWires = {ho_js};

function getCenter(id) {{
  const el = document.getElementById(id);
  if (!el) return {{x:0,y:0}};
  const r = el.getBoundingClientRect();
  const cr = document.getElementById('canvas').getBoundingClientRect();
  return {{ x: r.left - cr.left + r.width/2, y: r.top - cr.top + r.height/2 }};
}}

function qbp(t,x0,y0,cx,cy,x1,y1){{
  const m=1-t;
  return {{x:m*m*x0+2*m*t*cx+t*t*x1, y:m*m*y0+2*m*t*cy+t*t*y1}};
}}

function animateDot(svg,fx,fy,cx,cy,tx,ty,color,delay,r,phaseOffset) {{
  const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('r', r);
  c.setAttribute('fill', color);
  c.setAttribute('opacity', '0');
  svg.appendChild(c);
  let st = null;
  const dur = 1200 + phaseOffset * 200;
  function frame(ts) {{
    if (!st) st = ts + delay;
    const el = ts - st;
    if (el < 0) {{ requestAnimationFrame(frame); return; }}
    const t = (el % dur) / dur;
    const pos = qbp(t,fx,fy,cx,cy,tx,ty);
    c.setAttribute('cx', pos.x);
    c.setAttribute('cy', pos.y);
    const op = t<.08?t/.08 : t>.88?(1-t)/.12 : 1;
    c.setAttribute('opacity', op*0.9);
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}

function addArrow(defs, id, color) {{
  const mk = document.createElementNS('http://www.w3.org/2000/svg','marker');
  mk.setAttribute('id', id);
  mk.setAttribute('markerWidth','7');mk.setAttribute('markerHeight','7');
  mk.setAttribute('refX','5');mk.setAttribute('refY','3');
  mk.setAttribute('orient','auto');
  const pa = document.createElementNS('http://www.w3.org/2000/svg','path');
  pa.setAttribute('d','M0,0 L0,6 L7,3 z');
  pa.setAttribute('fill', color);
  mk.appendChild(pa); defs.appendChild(mk);
}}

function drawAll() {{
  const svg = document.getElementById('wires');
  const canvas = document.getElementById('canvas');
  svg.setAttribute('viewBox', `0 0 ${{canvas.offsetWidth}} ${{canvas.offsetHeight}}`);
  svg.innerHTML = '';
  const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
  svg.appendChild(defs);

  // phase 0 = input→hidden, phase 1 = hidden→output (delayed)
  function drawWireSet(wires, phaseDelay, curveSide) {{
    wires.forEach((w, i) => {{
      addArrow(defs, `arr_${{phaseDelay}}_${{i}}`, w.color);
      const from = getCenter(w.from);
      const to   = getCenter(w.to);
      const dx=to.x-from.x, dy=to.y-from.y, len=Math.sqrt(dx*dx+dy*dy);
      if(len===0) return;
      const fr=36, tr=42;
      const fx=from.x+dx/len*fr, fy=from.y+dy/len*fr;
      const tx=to.x  -dx/len*tr, ty=to.y  -dy/len*tr;
      const cx=(fx+tx)/2 + curveSide*dy*.15;
      const cy=(fy+ty)/2 - curveSide*dx*.15;
      const d=`M${{fx}},${{fy}} Q${{cx}},${{cy}} ${{tx}},${{ty}}`;

      // glow
      const g = document.createElementNS('http://www.w3.org/2000/svg','path');
      g.setAttribute('d',d); g.setAttribute('stroke',w.color);
      g.setAttribute('stroke-width',w.width+5); g.setAttribute('fill','none');
      g.setAttribute('opacity','0.12'); svg.appendChild(g);

      // line
      const l = document.createElementNS('http://www.w3.org/2000/svg','path');
      l.setAttribute('d',d); l.setAttribute('stroke',w.color);
      l.setAttribute('stroke-width',w.width); l.setAttribute('fill','none');
      l.setAttribute('opacity','0.8');
      l.setAttribute('marker-end', `url(#arr_${{phaseDelay}}_${{i}})`);
      svg.appendChild(l);

      // wire label
      const tag = document.createElement('div');
      tag.className='wire-tag';
      tag.textContent = w.label;
      tag.style.color = w.color;
      tag.style.left = (cx-22)+'px';
      tag.style.top  = (cy-10)+'px';
      document.getElementById('canvas').appendChild(tag);

      // traveling dots — 3 per wire, staggered
      const r = Math.max(2.5, w.width/1.4);
      for(let d2=0; d2<3; d2++) {{
        animateDot(svg, fx,fy,cx,cy,tx,ty,
                   w.color,
                   phaseDelay + d2*(1300/3),
                   r, i);
      }}
    }});
  }}

  drawWireSet(ihWires, 0,   1);   // input→hidden, dots start immediately
  drawWireSet(hoWires, 600, -1);  // hidden→output, dots start 600ms later
}}

window.addEventListener('load',  () => setTimeout(drawAll, 100));
window.addEventListener('resize', () => {{
  document.querySelectorAll('.wire-tag').forEach(t=>t.remove());
  drawAll();
}});
</script>
</body></html>"""


def show():
    st.title("➡️ Forward Propagation")
    st.markdown("""
    **Forward Propagation** is how a neural network makes a prediction —
    data flows **left to right**, layer by layer, until we get an output! 🚀

    Each neuron computes: `z = (weights × inputs) + bias`, then squishes it with an **activation function**.
    """)

    st.divider()

    # ── Config ────────────────────────────────────────────────────────
    st.subheader("🏗️ Network: 2 Inputs → 3 Hidden Neurons → 1 Output")

    col1, col2 = st.columns(2)
    with col1:
        activation = st.selectbox("Activation Function for Hidden Layer", ["Sigmoid", "ReLU"])
    with col2:
        st.markdown("""
        | Layer | Neurons | Role |
        |---|---|---|
        | Input | 2 | Receives data |
        | Hidden | 3 | Finds patterns |
        | Output | 1 | Makes prediction |
        """)

    st.divider()
    st.subheader("🎛️ Set Your Inputs — Watch the Network React!")

    col1, col2 = st.columns(2)
    with col1:
        x1 = st.slider("Input x1", 0.0, 1.0, 0.5, 0.01)
    with col2:
        x2 = st.slider("Input x2", 0.0, 1.0, 0.8, 0.01)

    # Fixed weights
    W1 = np.array([[0.2, 0.8], [0.4, 0.6], [-0.5, 0.9]])
    b1 = np.array([0.1, -0.1, 0.2])
    W2 = np.array([[0.3, -0.4, 0.7]])
    b2 = np.array([0.05])

    X      = np.array([x1, x2])
    act_fn = sigmoid if activation == "Sigmoid" else relu

    z1  = W1 @ X + b1
    a1  = act_fn(z1)
    z2  = W2 @ a1 + b2
    a2  = sigmoid(z2)

    # ── Animated diagram ──────────────────────────────────────────────
    html_code = forward_prop_html(x1, x2, a1, float(a2[0]), z1, float(z2[0]), activation)
    components.html(html_code, height=590, scrolling=False)

    # ── Step-by-step maths ────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Step-by-Step Calculation")

    with st.expander("🔍 Layer 1 — Hidden Layer (click to expand)", expanded=True):
        for i in range(3):
            color = "🟢" if a1[i] > 0 else "🔴"
            st.markdown(f"""
**{color} Neuron H{i+1}:**
| Step | Calculation | Result |
|---|---|---|
| Weighted sum | `{W1[i,0]:+.1f}×{x1:.2f} + {W1[i,1]:+.1f}×{x2:.2f} + {b1[i]:+.1f}` | **z = {z1[i]:+.4f}** |
| Activation | `{activation}({z1[i]:+.4f})` | **a = {a1[i]:.4f}** |
            """)

    with st.expander("🔍 Layer 2 — Output Layer (click to expand)", expanded=True):
        st.markdown(f"""
**🎯 Output Neuron:**
| Step | Calculation | Result |
|---|---|---|
| Weighted sum | `{W2[0,0]:+.1f}×{a1[0]:.3f} + {W2[0,1]:+.1f}×{a1[1]:.3f} + {W2[0,2]:+.1f}×{a1[2]:.3f} + {b2[0]:+.2f}` | **z = {z2[0]:+.4f}** |
| Activation | `sigmoid({z2[0]:+.4f})` | **output = {a2[0]:.4f}** |
| Decision | `{a2[0]:.4f} {"≥" if a2[0]>=0.5 else "<"} 0.5` | **{"Class 1 ✅" if a2[0]>=0.5 else "Class 0 ❌"}** |
        """)

    st.metric("🎯 Final Prediction", f"{a2[0]:.4f}",
              delta=f"{'Positive (≥0.5)' if a2[0]>=0.5 else 'Negative (<0.5)'}",
              help="Value between 0 and 1 — closer to 1 means more confident")

    # ── Output curve ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 How Output Changes as x1 Varies (x2 fixed)")

    x1_vals = np.linspace(0, 1, 150)
    outputs = []
    for v in x1_vals:
        Xv  = np.array([v, x2])
        z1v = W1 @ Xv + b1
        a1v = act_fn(z1v)
        z2v = W2 @ a1v + b2
        outputs.append(float(sigmoid(z2v)[0]))

    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="#2ecc71", opacity=0.05,
                  annotation_text="Predicts 1", annotation_position="top right")
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor="#e74c3c", opacity=0.05,
                  annotation_text="Predicts 0", annotation_position="bottom right")
    fig.add_trace(go.Scatter(
        x=x1_vals, y=outputs, mode="lines", name="Network Output",
        line=dict(color="#3498db", width=3),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)"
    ))
    fig.add_vline(x=x1, line_dash="dash", line_color="#ffd54f", line_width=2,
                  annotation_text=f"  x1={x1:.2f} → {a2[0]:.3f}",
                  annotation_font_color="#ffd54f")
    fig.add_hline(y=0.5, line_dash="dot", line_color="white", line_width=1,
                  opacity=0.4, annotation_text="  threshold=0.5",
                  annotation_font_color="white")
    fig.update_layout(
        title="Forward Pass Output vs x1",
        xaxis_title="x1 value", yaxis_title="Network Output",
        template="plotly_dark", yaxis_range=[0, 1],
        height=320,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Try switching between **Sigmoid** and **ReLU** above — notice how the curve shape changes!")