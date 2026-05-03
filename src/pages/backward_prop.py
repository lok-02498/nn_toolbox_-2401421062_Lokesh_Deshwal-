import streamlit as st
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json


# ─────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


# ─────────────────────────────────────────
# Animated backward-pass HTML diagram
# ─────────────────────────────────────────
def backprop_html(x1, x2, a1, a2_val, z1, z2_val,
                  delta2, delta1, dW2, dW1, target, loss):
    """
    Full-screen interactive backpropagation network diagram.
    Gradients flow RIGHT → LEFT, colored by sign (green = positive, red = negative).
    Each node shows both its forward-pass value AND its gradient.
    """

    # ── Color helpers ──────────────────────────────────────────────
    def grad_color_hex(g):
        return "#00e676" if g >= 0 else "#ff5252"

    def grad_rgba(g, alpha=0.85):
        if g >= 0:
            return f"rgba(0,230,118,{alpha})"
        return f"rgba(255,82,82,{alpha})"

    def node_glow(val, color_hex):
        s = min(1.0, max(0.0, abs(float(val)) * 2))
        return f"0 0 {int(10+s*28)}px {color_hex}, 0 0 {int(5+s*12)}px {color_hex}88"

    def act_rgba(val, base="230,126,34"):
        intensity = min(1.0, max(0.1, float(val)))
        alpha = 0.2 + intensity * 0.7
        return f"rgba({base},{alpha:.2f})"

    # ── Node display values ────────────────────────────────────────
    out_hex      = "#ffd54f"
    loss_str     = f"{loss:.5f}"
    a2_str       = f"{a2_val:.4f}"
    target_str   = f"{float(target):.2f}"
    delta2_str   = f"{float(delta2):+.4f}"
    delta2_color = grad_color_hex(float(delta2))

    # Hidden node colors driven by their forward activations
    h_colors = [act_rgba(v) for v in a1]
    # Gradient glow on hidden nodes driven by delta1
    d1_glows = [node_glow(v, grad_color_hex(v)) for v in delta1]
    d1_colors = [grad_color_hex(v) for v in delta1]

    # ── Wire data for backward pass ────────────────────────────────
    # output → hidden (3 wires, show dW2 gradients)
    oh_wires = []
    for i in range(2):
        g = float(dW2[i])
        oh_wires.append({
            "from": "out",
            "to": f"h{i}",
            "color": grad_color_hex(g),
            "width": max(1.5, abs(g) * 60),
            "label": f"∂L/∂W2={g:+.3f}",
            "signal": abs(g),
        })

    # hidden → input (4 wires, show dW1 gradients)
    hi_wires = []
    for i in range(2):
        for j in range(2):
            g = float(dW1[i, j])
            hi_wires.append({
                "from": f"h{i}",
                "to": f"in{j}",
                "color": grad_color_hex(g),
                "width": max(1.5, abs(g) * 60),
                "label": f"∂L/∂W1={g:+.3f}",
                "signal": abs(g),
            })

    oh_js = json.dumps(oh_wires)
    hi_js = json.dumps(hi_wires)

    a1_strs    = json.dumps([f"{v:.3f}" for v in a1])
    z1_strs    = json.dumps([f"{v:.3f}" for v in z1])
    d1_strs    = json.dumps([f"{v:+.4f}" for v in delta1])
    d1_col_js  = json.dumps(d1_colors)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:linear-gradient(135deg,#0d0d1f,#15102b,#0d0d1f);
  font-family:'Nunito',sans-serif;color:#e0e0e0;
  padding:14px;min-height:620px;
}}

/* ── TITLE ── */
.title{{font-family:'Fredoka One',cursive;font-size:20px;text-align:center;
        color:#ce93d8;margin-bottom:2px;letter-spacing:.5px}}
.subtitle{{text-align:center;font-size:11.5px;color:#7986cb;margin-bottom:10px}}

/* ── LOSS BANNER ── */
.loss-banner{{
  display:flex;align-items:center;justify-content:center;gap:16px;
  background:rgba(255,213,79,.07);border:1.5px solid rgba(255,213,79,.3);
  border-radius:12px;padding:8px 14px;margin-bottom:10px;
}}
.loss-banner .lbl{{font-family:'Fredoka One',cursive;font-size:12px;color:#90a4ae;text-transform:uppercase}}
.loss-banner .val{{font-family:'Fredoka One',cursive;font-size:20px;color:#ffd54f}}
.loss-banner .arrow{{font-size:22px;color:#ce93d8;animation:arrowpulse 1s ease-in-out infinite}}

/* ── CANVAS ── */
.canvas{{position:relative;width:100%;height:340px}}
svg.wires{{position:absolute;top:0;left:0;width:100%;height:100%;
           overflow:visible;pointer-events:none}}

/* ── NODES ── */
.node{{
  position:absolute;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  border-radius:50%;font-weight:800;text-align:center;
  transition:transform .3s;cursor:default;
}}
.node:hover{{transform:scale(1.16)!important;z-index:10}}

.node-in{{width:72px;height:72px;font-size:12px;border:2.5px solid #64b5f6;
          background:radial-gradient(circle at 35% 35%,rgba(52,152,219,.75),rgba(52,152,219,.25))}}
.node-h0{{width:78px;height:78px;font-size:11px;border:2.5px solid #ffb74d;
          animation:hpulse 2.2s ease-in-out infinite}}
.node-h1{{width:78px;height:78px;font-size:11px;border:2.5px solid #ffb74d;
          animation:hpulse 2.2s ease-in-out infinite .5s}}
.node-out{{width:86px;height:86px;font-size:12px;border:3px solid {out_hex};
           background:radial-gradient(circle at 35% 35%,{out_hex}cc,{out_hex}22);
           box-shadow:0 0 20px {out_hex}88}}
.node-loss{{width:72px;height:72px;font-size:11px;border:2.5px dashed #ce93d8;
            background:radial-gradient(circle at 35% 35%,rgba(206,147,216,.35),rgba(206,147,216,.08));
            animation:losspulse 1.5s ease-in-out infinite}}

.node-label{{font-size:9px;font-weight:700;opacity:.75;margin-top:1px;
             text-transform:uppercase;letter-spacing:.5px}}
.grad-badge{{
  position:absolute;background:rgba(10,5,25,.9);
  border-radius:6px;padding:2px 5px;font-size:9.5px;
  font-weight:800;white-space:nowrap;pointer-events:none;
  border:1px solid rgba(255,255,255,.15);
}}
.layer-label{{
  position:absolute;top:4px;
  font-family:'Fredoka One',cursive;font-size:12px;
  color:#546e7a;text-align:center;width:86px;
}}
.wire-tag{{
  position:absolute;background:rgba(5,5,20,.88);
  border:1px solid rgba(255,255,255,.18);
  border-radius:6px;padding:1px 5px;font-size:9px;
  font-weight:800;pointer-events:none;white-space:nowrap;
}}

/* ── INFO GRID ── */
.info-grid{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:7px;margin-top:10px}}
.info-card{{
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
  border-radius:11px;padding:9px 11px;backdrop-filter:blur(5px);
}}
.card-title{{font-family:'Fredoka One',cursive;font-size:11px;color:#78909c;
             margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px}}
.card-val{{font-family:'Fredoka One',cursive;font-size:15px}}
.card-sub{{font-size:9.5px;color:#607d8b;margin-top:2px}}

/* ── GLOSSARY PILL ROW ── */
.pills{{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}}
.pill{{
  background:rgba(206,147,216,.12);border:1px solid rgba(206,147,216,.3);
  border-radius:20px;padding:4px 10px;font-size:10.5px;font-weight:700;
  color:#ce93d8;cursor:default;
}}
.pill span{{color:#e0e0e0;font-weight:400}}

/* ── DIRECTION ARROW ── */
.dir-arrow{{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:26px;color:#ce93d8;opacity:.35;pointer-events:none;
  animation:arrowpulse 1.2s ease-in-out infinite;
}}

@keyframes hpulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.05)}}}}
@keyframes losspulse{{0%,100%{{transform:scale(1);opacity:1}}50%{{transform:scale(1.07);opacity:.8}}}}
@keyframes arrowpulse{{0%,100%{{opacity:.35}}50%{{opacity:.7}}}}
</style></head><body>

<div class="title">⬅️ Backward Propagation — Live Gradient Flow!</div>
<div class="subtitle">
  Gradients travel <strong style="color:#00e676">RIGHT → LEFT</strong> &nbsp;|&nbsp;
  <span style="color:#00e676">■</span> Positive gradient &nbsp;
  <span style="color:#ff5252">■</span> Negative gradient &nbsp;|&nbsp;
  Each node shows <em>activation</em> + <em>gradient (δ)</em>
</div>

<!-- Loss banner -->
<div class="loss-banner">
  <div><div class="lbl">Target (y)</div><div class="val" style="color:#42a5f5">{target_str}</div></div>
  <div class="arrow">→</div>
  <div><div class="lbl">Output (ŷ)</div><div class="val" style="color:{out_hex}">{a2_str}</div></div>
  <div class="arrow">→</div>
  <div><div class="lbl">Loss = ½(y-ŷ)²</div><div class="val" style="color:#ef9a9a">{loss_str}</div></div>
  <div class="arrow">⬅️</div>
  <div><div class="lbl">Backprop starts</div><div class="val" style="color:#ce93d8">here</div></div>
</div>

<div class="canvas" id="canvas">
  <svg class="wires" id="wires"></svg>

  <!-- Layer labels -->
  <div class="layer-label" style="left:2%">Input<br>Layer</div>
  <div class="layer-label" style="left:34%">Hidden<br>Layer</div>
  <div class="layer-label" style="left:64%">Output<br>Layer</div>
  <div class="layer-label" style="left:86%;color:#ce93d8">Loss<br>Node</div>

  <!-- ── Input nodes ── -->
  <div class="node node-in" id="in0"
       style="left:3%;top:28%;box-shadow:0 0 14px #3498db88">
    <div>x1</div>
    <div style="color:#ffd54f;font-size:14px">{x1:.2f}</div>
    <div class="node-label">Input 1</div>
  </div>
  <div class="node node-in" id="in1"
       style="left:3%;top:62%;box-shadow:0 0 14px #3498db88">
    <div>x2</div>
    <div style="color:#ffd54f;font-size:14px">{x2:.2f}</div>
    <div class="node-label">Input 2</div>
  </div>

  <!-- ── Hidden nodes ── -->
  <div class="node node-h0" id="h0"
       style="left:33%;top:20%;background:radial-gradient(circle at 35% 35%,{h_colors[0]},{h_colors[0]});box-shadow:{d1_glows[0]}">
    <div style="font-size:9px;color:#b0bec5">a={float(a1[0]):.3f}</div>
    <div style="color:{d1_colors[0]};font-size:13px">δ={float(delta1[0]):+.3f}</div>
    <div class="node-label">H1</div>
  </div>
  <div class="node node-h1" id="h1"
       style="left:33%;top:60%;background:radial-gradient(circle at 35% 35%,{h_colors[1]},{h_colors[1]});box-shadow:{d1_glows[1]}">
    <div style="font-size:9px;color:#b0bec5">a={float(a1[1]):.3f}</div>
    <div style="color:{d1_colors[1]};font-size:13px">δ={float(delta1[1]):+.3f}</div>
    <div class="node-label">H2</div>
  </div>

  <!-- ── Output node ── -->
  <div class="node node-out" id="out"
       style="left:63%;top:40%">
    <div style="font-size:9px;color:#b0bec5">ŷ={a2_str}</div>
    <div style="color:{delta2_color};font-size:12px">δ={delta2_str}</div>
    <div class="node-label">Output</div>
  </div>

  <!-- ── Loss node ── -->
  <div class="node node-loss" id="loss_node"
       style="left:85%;top:40%">
    <div style="font-size:9px;color:#b0bec5">L=</div>
    <div style="color:#ef9a9a;font-size:13px">{loss_str}</div>
    <div class="node-label">Loss</div>
  </div>

</div><!-- end canvas -->

<!-- Info cards -->
<div class="info-grid">
  <div class="info-card">
    <div class="card-title">📉 Loss</div>
    <div class="card-val" style="color:#ef9a9a">{loss_str}</div>
    <div class="card-sub">½×(y−ŷ)² measures error</div>
  </div>
  <div class="info-card">
    <div class="card-title">🎯 Output δ</div>
    <div class="card-val" style="color:{delta2_color}">{delta2_str}</div>
    <div class="card-sub">(ŷ−y)×σ′(z₂)</div>
  </div>
  <div class="info-card">
    <div class="card-title">🟠 Hidden δ</div>
    <div class="card-val" style="color:#ffa726">[{float(delta1[0]):+.2f}, {float(delta1[1]):+.2f}]</div>
    <div class="card-sub">W2ᵀ×δ_out × σ′(z₁)</div>
  </div>
  <div class="info-card">
    <div class="card-title">📐 ∂L/∂W2</div>
    <div class="card-val" style="color:#00e676">[{float(dW2[0]):+.3f}, {float(dW2[1]):+.3f}]</div>
    <div class="card-sub">δ_out × a_hidden</div>
  </div>
</div>

<!-- Glossary pills -->
<div class="pills">
  <div class="pill">δ (delta) <span>= local gradient at a neuron</span></div>
  <div class="pill">σ′(z) <span>= sigmoid derivative</span></div>
  <div class="pill">∂L/∂W <span>= weight gradient (how to nudge W)</span></div>
  <div class="pill">α <span>= learning rate</span></div>
  <div class="pill">W_new = W − α·∂L/∂W <span>= gradient descent step</span></div>
</div>

<script>
const ohWires = {oh_js};
const hiWires = {hi_js};

function getCenter(id) {{
  const el = document.getElementById(id);
  if (!el) return {{x:0,y:0}};
  const r  = el.getBoundingClientRect();
  const cr = document.getElementById('canvas').getBoundingClientRect();
  return {{ x: r.left - cr.left + r.width/2, y: r.top - cr.top + r.height/2 }};
}}

function qbp(t,x0,y0,cx,cy,x1,y1){{
  const m=1-t;
  return {{x:m*m*x0+2*m*t*cx+t*t*x1, y:m*m*y0+2*m*t*cy+t*t*y1}};
}}

function animateDot(svg,fx,fy,cx,cy,tx,ty,color,delay,r,phaseOffset){{
  const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('r', r);
  c.setAttribute('fill', color);
  c.setAttribute('opacity','0');
  svg.appendChild(c);
  let st = null;
  const dur = 1100 + phaseOffset*180;
  function frame(ts){{
    if(!st) st = ts + delay;
    const el = ts - st;
    if(el < 0){{requestAnimationFrame(frame);return;}}
    const t = (el % dur) / dur;
    // Dots travel from 'to' back to 'from' (right→left = backward)
    const pos = qbp(t,tx,ty,cx,cy,fx,fy);
    c.setAttribute('cx', pos.x);
    c.setAttribute('cy', pos.y);
    const op = t<.08?t/.08 : t>.88?(1-t)/.12 : 1;
    c.setAttribute('opacity', op*0.85);
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}

function addArrow(defs,id,color){{
  const mk = document.createElementNS('http://www.w3.org/2000/svg','marker');
  mk.setAttribute('id',id);
  mk.setAttribute('markerWidth','7');mk.setAttribute('markerHeight','7');
  mk.setAttribute('refX','2');mk.setAttribute('refY','3');
  mk.setAttribute('orient','auto');
  const pa = document.createElementNS('http://www.w3.org/2000/svg','path');
  pa.setAttribute('d','M7,0 L7,6 L0,3 z'); // flipped arrow (pointing left)
  pa.setAttribute('fill',color);
  mk.appendChild(pa);defs.appendChild(mk);
}}

function lossToOutWire(svg, defs){{
  // Special wire: loss_node → out (shows gradient origin)
  addArrow(defs,'arr_loss','#ce93d8');
  const from = getCenter('loss_node');
  const to   = getCenter('out');
  const dx=to.x-from.x, dy=to.y-from.y, len=Math.sqrt(dx*dx+dy*dy);
  if(len===0) return;
  const fr=36,tr=43;
  const fx=from.x+dx/len*fr, fy=from.y+dy/len*fr;
  const tx=to.x  -dx/len*tr, ty=to.y  -dy/len*tr;
  const cx=(fx+tx)/2, cy=(fy+ty)/2;
  const d=`M${{fx}},${{fy}} L${{tx}},${{ty}}`;
  const g = document.createElementNS('http://www.w3.org/2000/svg','path');
  g.setAttribute('d',d);g.setAttribute('stroke','#ce93d8');
  g.setAttribute('stroke-width','3');g.setAttribute('fill','none');
  g.setAttribute('opacity','0.15');svg.appendChild(g);
  const l = document.createElementNS('http://www.w3.org/2000/svg','path');
  l.setAttribute('d',d);l.setAttribute('stroke','#ce93d8');
  l.setAttribute('stroke-width','2');l.setAttribute('fill','none');
  l.setAttribute('opacity','0.7');l.setAttribute('stroke-dasharray','5,3');
  l.setAttribute('marker-start','url(#arr_loss)');
  svg.appendChild(l);
  for(let d2=0;d2<2;d2++){{
    animateDot(svg,fx,fy,cx,cy,tx,ty,'#ce93d8',d2*600,3,0);
  }}
}}

function drawWireSet(svg,defs,wires,phaseDelay,curveSide,pfx){{
  wires.forEach((w,i)=>{{
    addArrow(defs,`arr_${{pfx}}_${{i}}`,w.color);
    const from = getCenter(w.from);
    const to   = getCenter(w.to);
    const dx=to.x-from.x,dy=to.y-from.y,len=Math.sqrt(dx*dx+dy*dy);
    if(len===0) return;
    const fr=43,tr=36;
    const fx=from.x+dx/len*fr, fy=from.y+dy/len*fr;
    const tx2=to.x  -dx/len*tr, ty2=to.y  -dy/len*tr;
    const cx=(fx+tx2)/2 + curveSide*dy*.15;
    const cy=(fy+ty2)/2 - curveSide*dx*.15;
    const d=`M${{fx}},${{fy}} Q${{cx}},${{cy}} ${{tx2}},${{ty2}}`;

    // glow path
    const g2 = document.createElementNS('http://www.w3.org/2000/svg','path');
    g2.setAttribute('d',d);g2.setAttribute('stroke',w.color);
    g2.setAttribute('stroke-width',w.width+6);g2.setAttribute('fill','none');
    g2.setAttribute('opacity','0.1');svg.appendChild(g2);

    // main path
    const l = document.createElementNS('http://www.w3.org/2000/svg','path');
    l.setAttribute('d',d);l.setAttribute('stroke',w.color);
    l.setAttribute('stroke-width',w.width);l.setAttribute('fill','none');
    l.setAttribute('opacity','0.8');
    l.setAttribute('marker-end',`url(#arr_${{pfx}}_${{i}})`);
    svg.appendChild(l);

    // wire label
    const tag = document.createElement('div');
    tag.className='wire-tag';
    tag.textContent=w.label;
    tag.style.color=w.color;
    tag.style.left=(cx-30)+'px';
    tag.style.top=(cy-11)+'px';
    document.getElementById('canvas').appendChild(tag);

    // traveling dots (backward direction: from "from" toward "to" but we reverse in animateDot)
    const r2=Math.max(2.5,w.width/1.3);
    for(let d3=0;d3<3;d3++){{
      animateDot(svg,fx,fy,cx,cy,tx2,ty2,
                 w.color, phaseDelay+d3*(1100/3), r2, i);
    }}
  }});
}}

function drawAll(){{
  const svg    = document.getElementById('wires');
  const canvas = document.getElementById('canvas');
  svg.setAttribute('viewBox',`0 0 ${{canvas.offsetWidth}} ${{canvas.offsetHeight}}`);
  svg.innerHTML='';
  document.querySelectorAll('.wire-tag').forEach(t=>t.remove());
  const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
  svg.appendChild(defs);

  lossToOutWire(svg, defs);                          // loss → output (start)
  drawWireSet(svg,defs,ohWires,  0, -1,'oh');        // output → hidden
  drawWireSet(svg,defs,hiWires,700,  1,'hi');        // hidden → input (delayed)
}}

window.addEventListener('load', ()=>setTimeout(drawAll,120));
window.addEventListener('resize',()=>{{
  document.querySelectorAll('.wire-tag').forEach(t=>t.remove());
  drawAll();
}});
</script>
</body></html>"""


# ─────────────────────────────────────────
# Main Streamlit page
# ─────────────────────────────────────────
def show():
    st.title("⬅️ Backward Propagation")
    st.markdown("""
    **Backpropagation** is how a neural network **learns from its mistakes**.

    Think of it like this: after the network makes a wrong guess, we need to figure out  
    *"which weights were most responsible?"* and nudge them in the right direction.  
    That nudging process — computed backwards from the error — is backprop! 🔁
    """)

    # ── Beginner glossary ──────────────────────────────────────────
    with st.expander("📖 Beginner Glossary — What do all these symbols mean?", expanded=False):
        st.markdown("""
| Symbol | Name | Plain English |
|--------|------|---------------|
| **ŷ** | Prediction | What the network guessed |
| **y** | Target | What the correct answer actually is |
| **L** | Loss | How wrong the network was (lower = better) |
| **δ (delta)** | Local gradient | "How much is this neuron responsible for the error?" |
| **σ′(z)** | Sigmoid derivative | How sensitive the neuron's output is to small changes |
| **∂L/∂W** | Weight gradient | "Which direction should I move this weight to reduce loss?" |
| **α (alpha)** | Learning rate | How big a step to take when updating weights |
| **W_new = W − α·∂L/∂W** | Gradient descent | The actual weight update formula |

> 💡 **Key insight:** We use the **chain rule** from calculus to multiply gradients layer by layer, all the way back to the first weights.
        """)

    st.divider()

    # ── Controls ───────────────────────────────────────────────────
    st.subheader("🎛️ Set Up the Problem")
    col1, col2, col3 = st.columns(3)
    with col1:
        x1 = st.slider("Input x1", 0.0, 1.0, 0.5, 0.01,
                        help="First feature fed into the network")
    with col2:
        x2 = st.slider("Input x2", 0.0, 1.0, 0.3, 0.01,
                        help="Second feature fed into the network")
    with col3:
        target = st.slider("Target Output (y)", 0.0, 1.0, 1.0, 0.01,
                           help="The correct answer we want the network to output")

    lr = st.slider("📏 Learning Rate (α)", 0.01, 2.0, 0.5, 0.01,
                   help="Controls how big a step we take in the direction of the gradient")

    # Fixed weights (reproducible demo)
    W1 = np.array([[0.15, 0.25], [0.20, 0.30]])
    b1 = np.array([0.35, 0.35])
    W2 = np.array([[0.40, 0.50]])
    b2 = np.array([0.60])

    X = np.array([x1, x2])

    # ── Forward pass ───────────────────────────────────────────────
    z1  = W1 @ X + b1
    a1  = sigmoid(z1)
    z2  = W2 @ a1 + b2
    a2  = sigmoid(z2)
    loss = 0.5 * (target - a2[0])**2

    # ── Backward pass ──────────────────────────────────────────────
    dL_da2  = -(target - a2[0])          # ∂L/∂a2
    da2_dz2 = sigmoid_deriv(z2[0])       # σ′(z2)
    delta2  = dL_da2 * da2_dz2           # output layer delta

    dW2 = delta2 * a1                    # gradient for W2
    db2 = delta2                         # gradient for b2

    da1_dz1 = sigmoid_deriv(z1)          # σ′(z1) for each hidden neuron
    delta1  = (W2[0] * delta2) * da1_dz1 # hidden layer deltas

    dW1 = np.outer(delta1, X)            # gradient for W1
    db1 = delta1                         # gradient for b1

    # ── Updated weights ────────────────────────────────────────────
    W1_new = W1 - lr * dW1
    b1_new = b1 - lr * db1
    W2_new = W2 - lr * dW2
    b2_new = b2 - lr * db2

    # New loss after one update
    z1n   = W1_new @ X + b1_new
    a1n   = sigmoid(z1n)
    z2n   = W2_new @ a1n + b2_new
    a2n   = sigmoid(z2n)
    new_loss = 0.5 * (target - a2n[0])**2

    # ── Animated diagram ───────────────────────────────────────────
    st.divider()
    st.subheader("🔁 Live Network — Gradient Flow Animated")
    st.caption("Glowing wires show gradients flowing backward (right → left). "
               "Dot speed = gradient magnitude. Green = positive, Red = negative.")

    html_code = backprop_html(x1, x2, a1, float(a2[0]), z1, float(z2[0]),
                               delta2, delta1, dW2, dW1, target, loss)
    components.html(html_code, height=640, scrolling=False)

    # ── Step-by-step walkthrough ────────────────────────────────────
    st.divider()
    st.subheader("📊 Step-by-Step Calculation (Node by Node)")

    # STEP 1
    with st.expander("➡️ STEP 1 — Forward Pass (quick recap)", expanded=False):
        st.markdown("*Before we can go backward, we must know what the network predicted.*")
        col1, col2, col3 = st.columns(3)
        col1.metric("Hidden H1", f"{a1[0]:.4f}", help=f"sigmoid({z1[0]:.4f})")
        col2.metric("Hidden H2", f"{a1[1]:.4f}", help=f"sigmoid({z1[1]:.4f})")
        col3.metric("Output ŷ",  f"{a2[0]:.4f}", help=f"sigmoid({z2[0]:.4f})")
        st.metric("📉 Loss (MSE)", f"{loss:.6f}",
                  help="½ × (target − output)² — lower is better")

    # STEP 2
    with st.expander("⬅️ STEP 2 — Output Layer Gradient", expanded=True):
        st.markdown("""
**Goal:** How much did the output neuron contribute to the loss?

We use the **chain rule**:
```
δ_output = ∂L/∂ŷ  ×  σ′(z_output)
         = (ŷ − y) × σ′(z_output)
```
        """)
        st.markdown(f"""
| Term | Formula | Value |
|------|---------|-------|
| Error signal ∂L/∂ŷ | `ŷ − y = {a2[0]:.4f} − {target:.2f}` | **{dL_da2:+.4f}** |
| Sigmoid derivative σ′(z₂) | `σ({z2[0]:.4f}) × (1 − σ({z2[0]:.4f}))` | **{da2_dz2:.4f}** |
| **Output delta δ** | `{dL_da2:+.4f} × {da2_dz2:.4f}` | **{delta2:+.4f}** |

> 🟢 **Positive δ** → output was *too high* → we'll push weights **down**  
> 🔴 **Negative δ** → output was *too low*  → we'll push weights **up**
        """)
        col1, col2 = st.columns(2)
        col1.markdown(f"""
**Weight gradients ∂L/∂W2:**
| Weight | Gradient | Meaning |
|--------|---------|---------|
| W2[H1→Out] | `{delta2:+.4f} × {a1[0]:.4f} = {dW2[0]:+.4f}` | nudge by `{-lr*dW2[0]:+.4f}` |
| W2[H2→Out] | `{delta2:+.4f} × {a1[1]:.4f} = {dW2[1]:+.4f}` | nudge by `{-lr*dW2[1]:+.4f}` |
        """)
        col2.markdown(f"""
**Bias gradient ∂L/∂b2:**
`δ_output = {delta2:+.4f}`  
→ bias nudged by `{-lr*float(db2):+.4f}`
        """)

    # STEP 3
    with st.expander("⬅️ STEP 3 — Hidden Layer Gradients", expanded=True):
        st.markdown("""
**Goal:** How much did each hidden neuron contribute?

We "send back" the output delta through the weights, then multiply by the neuron's own sigmoid derivative:
```
δ_hidden[i] = W2[i] × δ_output  ×  σ′(z_hidden[i])
```
        """)
        for i in range(2):
            d1_color = "🟢" if delta1[i] >= 0 else "🔴"
            st.markdown(f"""
**{d1_color} Hidden Neuron H{i+1}:**
| Term | Value |
|------|-------|
| Incoming weight W2[{i}] | `{W2[0,i]:.2f}` |
| Output delta δ_out | `{delta2:+.4f}` |
| Sigmoid deriv σ′(z₁[{i}]) | `{da1_dz1[i]:.4f}` |
| **Hidden delta δ_h{i+1}** | `{W2[0,i]:.2f} × {delta2:+.4f} × {da1_dz1[i]:.4f} = {delta1[i]:+.4f}` |
            """)

        st.markdown(f"""
**Weight gradients ∂L/∂W1:**

| Weight | ∂L/∂W = δ_hidden × input | Value |
|--------|--------------------------|-------|
| W1[H1←x1] | `{delta1[0]:+.4f} × {x1:.4f}` | **{dW1[0,0]:+.4f}** |
| W1[H1←x2] | `{delta1[0]:+.4f} × {x2:.4f}` | **{dW1[0,1]:+.4f}** |
| W1[H2←x1] | `{delta1[1]:+.4f} × {x1:.4f}` | **{dW1[1,0]:+.4f}** |
| W1[H2←x2] | `{delta1[1]:+.4f} × {x2:.4f}` | **{dW1[1,1]:+.4f}** |
        """)

    # STEP 4
    with st.expander("🔄 STEP 4 — Weight Update (Gradient Descent)", expanded=True):
        st.markdown(f"""
**Formula:** `W_new = W − α × ∂L/∂W`  
with learning rate **α = {lr}**

**W2 (output layer weights):**
        """)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Before:**\n`[{W2[0,0]:.4f}, {W2[0,1]:.4f}]`")
        col2.markdown(f"**Δ (−α·grad):**\n`[{-lr*dW2[0]:+.4f}, {-lr*dW2[1]:+.4f}]`")
        col3.markdown(f"**After:**\n`[{W2_new[0,0]:.4f}, {W2_new[0,1]:.4f}]`")

        st.markdown("**W1 (hidden layer weights):**")
        st.markdown(f"""
| Weight | Before | Δ | After |
|--------|--------|---|-------|
| W1[0,0] | `{W1[0,0]:.4f}` | `{-lr*dW1[0,0]:+.4f}` | **`{W1_new[0,0]:.4f}`** |
| W1[0,1] | `{W1[0,1]:.4f}` | `{-lr*dW1[0,1]:+.4f}` | **`{W1_new[0,1]:.4f}`** |
| W1[1,0] | `{W1[1,0]:.4f}` | `{-lr*dW1[1,0]:+.4f}` | **`{W1_new[1,0]:.4f}`** |
| W1[1,1] | `{W1[1,1]:.4f}` | `{-lr*dW1[1,1]:+.4f}` | **`{W1_new[1,1]:.4f}`** |
        """)

    # Result
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Old Loss",    f"{loss:.6f}")
    col2.metric("New Loss",    f"{new_loss:.6f}", delta=f"{new_loss - loss:.6f}")
    col3.metric("Improvement", f"{(loss - new_loss):.6f}",
                delta="✅ Better!" if new_loss < loss else "⚠️ Worse")

    if new_loss < loss:
        st.success("✅ Loss decreased — the network learned something this step!")
    else:
        st.warning("⚠️ Loss increased — try a smaller learning rate (α).")

    # ── Training curve ─────────────────────────────────────────────
    st.divider()
    st.subheader("📉 Loss Over 100 Training Iterations")
    st.caption("Watch how the loss curve converges — each dip is one backprop + weight update step.")

    W1t, b1t, W2t, b2t = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    losses, outputs_over_time = [], []
    for _ in range(100):
        z1t = W1t @ X + b1t
        a1t = sigmoid(z1t)
        z2t = W2t @ a1t + b2t
        a2t = sigmoid(z2t)
        lt  = 0.5 * (target - a2t[0])**2
        losses.append(lt)
        outputs_over_time.append(float(a2t[0]))

        d2  = -(target - a2t[0]) * sigmoid_deriv(z2t[0])
        W2t -= lr * d2 * a1t
        b2t -= lr * d2
        d1  = (W2t[0] * d2) * sigmoid_deriv(z1t)
        W1t -= lr * np.outer(d1, X)
        b1t -= lr * d1

    fig = go.Figure()

    # Shaded regions
    fig.add_hrect(y0=0, y1=losses[0]*0.05, fillcolor="#2ecc71",
                  opacity=0.08, annotation_text="Near zero loss", annotation_position="right")

    fig.add_trace(go.Scatter(
        y=losses, mode="lines", name="Loss",
        line=dict(color="#ef5350", width=3),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.08)"
    ))
    fig.add_trace(go.Scatter(
        y=outputs_over_time, mode="lines", name="Output (ŷ)",
        line=dict(color="#ffd54f", width=2, dash="dot"),
        yaxis="y2"
    ))
    fig.add_hline(y=target, line_dash="dash", line_color="#42a5f5",
                  line_width=1.5, annotation_text=f"  target={target:.2f}",
                  annotation_font_color="#42a5f5")

    fig.update_layout(
        title="Loss & Output Converging Toward Target",
        xaxis_title="Training Iteration",
        yaxis_title="Loss",
        yaxis2=dict(title="Output (ŷ)", overlaying="y", side="right",
                    range=[0, 1], showgrid=False),
        template="plotly_dark",
        height=340,
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Gradient magnitude heatmap ──────────────────────────────────
    st.divider()
    st.subheader("🌡️ Gradient Magnitude Heatmap — Which Weights Changed Most?")
    st.caption("Brighter = larger gradient = that weight had the biggest influence on the error.")

    all_grads = np.array([
        [abs(dW1[0,0]), abs(dW1[0,1])],
        [abs(dW1[1,0]), abs(dW1[1,1])],
    ])
    w2_grads = np.array([[abs(dW2[0]), abs(dW2[1])]])

    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(
        z=all_grads,
        x=["x1", "x2"],
        y=["H1", "H2"],
        colorscale="Plasma",
        showscale=True,
        name="W1 Gradients",
        text=[[f"{v:.4f}" for v in row] for row in all_grads],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorbar=dict(title="∂L/∂W", x=0.45),
    ))
    fig2.add_trace(go.Heatmap(
        z=w2_grads,
        x=["H1→Out", "H2→Out"],
        y=["Out"],
        colorscale="Plasma",
        showscale=False,
        xaxis="x2", yaxis="y2",
        text=[[f"{v:.4f}" for v in row] for row in w2_grads],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
    ))
    fig2.update_layout(
        template="plotly_dark",
        height=260,
        title="W1 Gradients (left) | W2 Gradients (right)",
        xaxis=dict(domain=[0, 0.45]),
        xaxis2=dict(domain=[0.55, 1.0]),
        yaxis2=dict(anchor="x2"),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
💡 **Key takeaways for beginners:**  
1. **Backprop = chain rule applied repeatedly** — gradient at each layer depends on the layer in front of it.  
2. **Large gradient → large weight change** — weights that contributed more to the error get updated more.  
3. **Learning rate too large?** Loss explodes. **Too small?** Learning is painfully slow.  
4. **Every training step** does one forward pass + one backward pass + one weight update.
    """)