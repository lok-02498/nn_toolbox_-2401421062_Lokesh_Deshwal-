import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit.components.v1 as components


def step(x):
    return 1 if x >= 0 else 0


def animated_perceptron_html(x1, x2, w1, w2, bias, z, output):
    out_color   = "#00e676" if output == 1 else "#ff5252"
    out_emoji   = "✅ ON!"  if output == 1 else "❌ OFF"
    w1_color    = "#00e676" if w1 >= 0 else "#ff5252"
    w2_color    = "#00e676" if w2 >= 0 else "#ff5252"
    w1_thick    = max(2, int(abs(w1) * 5))
    w2_thick    = max(2, int(abs(w2) * 5))
    z_sign      = ">= 0 fires!" if z >= 0 else "< 0 silent"
    verdict     = "Neuron FIRES! Signal gets through!" if output == 1 else "Neuron is QUIET. No signal."
    verdict_pre = "🔥" if output == 1 else "💤"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:linear-gradient(135deg,#0f0c29,#1a1a4e,#0f0c29);font-family:'Nunito',sans-serif;color:white;padding:14px;min-height:510px}}
.title{{font-family:'Fredoka One',cursive;font-size:21px;text-align:center;color:#ffd54f;margin-bottom:8px;letter-spacing:1px}}
.canvas{{position:relative;width:100%;height:310px}}
svg.wires{{position:absolute;top:0;left:0;width:100%;height:100%;overflow:visible}}
.node{{position:absolute;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:50%;font-weight:800;cursor:default;transition:transform .3s ease,box-shadow .3s ease;text-align:center}}
.node:hover{{transform:scale(1.15)!important}}
.node-input{{width:76px;height:76px;background:radial-gradient(circle at 35% 35%,#42a5f5,#1565c0);box-shadow:0 0 20px #42a5f588,0 4px 16px #000a;font-size:14px;border:3px solid #90caf9}}
.node-bias{{width:66px;height:66px;background:radial-gradient(circle at 35% 35%,#ce93d8,#6a1b9a);box-shadow:0 0 18px #ce93d888,0 4px 16px #000a;font-size:12px;border:3px solid #e040fb}}
.node-neuron{{width:92px;height:92px;background:radial-gradient(circle at 35% 35%,#ffb74d,#e65100);box-shadow:0 0 40px #ff9800,0 0 80px #ff980055;font-size:26px;border:3px solid #ffcc02;animation:pulse 1.6s ease-in-out infinite}}
.node-output{{width:82px;height:82px;background:radial-gradient(circle at 35% 35%,{out_color}cc,{out_color}44);box-shadow:0 0 30px {out_color}99,0 0 60px {out_color}44;font-size:12px;font-weight:800;border:3px solid {out_color};animation:outputPop .6s cubic-bezier(.36,.07,.19,.97) both}}
.node-label{{font-size:9px;font-weight:700;opacity:.85;margin-top:2px;text-transform:uppercase;letter-spacing:.5px}}
.wire-tag{{position:absolute;background:rgba(10,10,30,.88);border:1.5px solid #ffffff33;border-radius:8px;padding:2px 7px;font-size:12px;font-weight:800;pointer-events:none;white-space:nowrap;backdrop-filter:blur(4px)}}
.info-panel{{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);border-radius:14px;padding:12px 16px;margin-top:8px;backdrop-filter:blur(8px)}}
.info-row{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.07);font-size:13px}}
.info-row:last-child{{border-bottom:none}}
.info-key{{color:#b0bec5;font-weight:700}}
.info-val{{font-weight:800;font-family:'Fredoka One',cursive;font-size:14px}}
.verdict{{text-align:center;margin-top:9px;font-family:'Fredoka One',cursive;font-size:19px;animation:bounceIn .5s ease both}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 40px #ff9800,0 0 80px #ff980055;transform:scale(1)}}50%{{box-shadow:0 0 55px #ff9800,0 0 110px #ff980044;transform:scale(1.05)}}}}
@keyframes outputPop{{0%{{transform:scale(.5);opacity:0}}70%{{transform:scale(1.15);opacity:1}}100%{{transform:scale(1)}}}}
@keyframes bounceIn{{0%{{transform:scale(.3);opacity:0}}60%{{transform:scale(1.1)}}100%{{transform:scale(1);opacity:1}}}}
</style></head><body>
<div class="title">🧠 How a Perceptron Thinks!</div>
<div class="canvas" id="canvas">
  <svg class="wires" id="wires"></svg>
  <div class="node node-input" id="n-x1" style="left:3%;top:18%;">
    <div>x1=<b>{x1}</b></div><div class="node-label">Input 1</div>
  </div>
  <div class="node node-input" id="n-x2" style="left:3%;top:55%;">
    <div>x2=<b>{x2}</b></div><div class="node-label">Input 2</div>
  </div>
  <div class="node node-bias" id="n-bias" style="left:6%;top:85%;">
    <div style="font-size:11px"><b>{bias:+.1f}</b></div><div class="node-label">Bias</div>
  </div>
  <div class="node node-neuron" id="n-neuron" style="left:37%;top:32%;">
    <div>&#x03A3;</div><div class="node-label" style="font-size:8px">Adder</div>
  </div>
  <div class="node node-output" id="n-output" style="left:73%;top:32%;">
    <div style="font-size:16px;">{out_emoji}</div><div class="node-label">Output</div>
  </div>
</div>
<div class="info-panel">
  <div class="info-row">
    <span class="info-key">🔢 Neuron calculates:</span>
    <span class="info-val" style="color:#ffd54f;">({w1:+.1f})x{x1} + ({w2:+.1f})x{x2} + ({bias:+.1f}) = <span style="color:#ff9800">{z:+.3f}</span></span>
  </div>
  <div class="info-row">
    <span class="info-key">⚡ z={z:+.3f} is {z_sign}</span>
    <span class="info-val" style="color:{out_color};">Output = {output}</span>
  </div>
  <div class="info-row">
    <span class="info-key">📏 Rule:</span>
    <span class="info-val" style="font-size:12px;color:#b0bec5;">z&ge;0 &rarr; <span style="color:#00e676">Fire!(1)</span> &nbsp;|&nbsp; z&lt;0 &rarr; <span style="color:#ff5252">Silent(0)</span></span>
  </div>
</div>
<div class="verdict" style="color:{out_color};">{verdict_pre} {verdict}</div>
<script>
function getCenter(el){{const r=el.getBoundingClientRect();const cr=document.getElementById('canvas').getBoundingClientRect();return{{x:r.left-cr.left+r.width/2,y:r.top-cr.top+r.height/2}}}}
const wires=[
  {{from:'n-x1',to:'n-neuron',color:'{w1_color}',width:{w1_thick},label:'w1={w1:+.1f}',dash:false}},
  {{from:'n-x2',to:'n-neuron',color:'{w2_color}',width:{w2_thick},label:'w2={w2:+.1f}',dash:false}},
  {{from:'n-bias',to:'n-neuron',color:'#ce93d8',width:2,label:'b={bias:+.1f}',dash:true}},
  {{from:'n-neuron',to:'n-output',color:'{out_color}',width:4,label:'z={z:+.2f}',dash:false}}
];
function drawWires(){{
  const svg=document.getElementById('wires');
  const canvas=document.getElementById('canvas');
  svg.setAttribute('viewBox',`0 0 ${{canvas.offsetWidth}} ${{canvas.offsetHeight}}`);
  svg.innerHTML='';
  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs');
  const filt=document.createElementNS('http://www.w3.org/2000/svg','filter');
  filt.setAttribute('id','dotBlur');
  const blur=document.createElementNS('http://www.w3.org/2000/svg','feGaussianBlur');
  blur.setAttribute('stdDeviation','1.5');
  filt.appendChild(blur);
  defs.appendChild(filt);
  wires.forEach((w,i)=>{{
    const mk=document.createElementNS('http://www.w3.org/2000/svg','marker');
    mk.setAttribute('id',`arrow${{i}}`);mk.setAttribute('markerWidth','8');mk.setAttribute('markerHeight','8');
    mk.setAttribute('refX','6');mk.setAttribute('refY','3');mk.setAttribute('orient','auto');
    const pa=document.createElementNS('http://www.w3.org/2000/svg','path');
    pa.setAttribute('d','M0,0 L0,6 L8,3 z');pa.setAttribute('fill',w.color);
    mk.appendChild(pa);defs.appendChild(mk);
  }});
  svg.appendChild(defs);
  wires.forEach((w,i)=>{{
    const from=getCenter(document.getElementById(w.from));
    const to=getCenter(document.getElementById(w.to));
    const dx=to.x-from.x,dy=to.y-from.y,len=Math.sqrt(dx*dx+dy*dy);
    const fr=38,tr=46;
    const fx=from.x+dx/len*fr,fy=from.y+dy/len*fr;
    const tx=to.x-dx/len*tr,ty=to.y-dy/len*tr;
    const cx=(fx+tx)/2-dy*.12,cy=(fy+ty)/2+dx*.12;
    const d=`M${{fx}},${{fy}} Q${{cx}},${{cy}} ${{tx}},${{ty}}`;
    const glow=document.createElementNS('http://www.w3.org/2000/svg','path');
    glow.setAttribute('d',d);glow.setAttribute('stroke',w.color);
    glow.setAttribute('stroke-width',w.width+6);glow.setAttribute('fill','none');
    glow.setAttribute('opacity','0.15');glow.setAttribute('filter','blur(4px)');
    svg.appendChild(glow);
    const line=document.createElementNS('http://www.w3.org/2000/svg','path');
    line.setAttribute('d',d);line.setAttribute('stroke',w.color);
    line.setAttribute('stroke-width',w.width);line.setAttribute('fill','none');
    line.setAttribute('opacity','0.9');
    if(w.dash)line.setAttribute('stroke-dasharray','7,5');
    line.setAttribute('marker-end',`url(#arrow${{i}})`);
    svg.appendChild(line);
    const tag=document.createElement('div');
    tag.className='wire-tag';tag.textContent=w.label;tag.style.color=w.color;
    tag.style.left=(cx-28)+'px';tag.style.top=(cy-11)+'px';
    document.getElementById('canvas').appendChild(tag);
    for(let d2=0;d2<3;d2++)animateDot(svg,fx,fy,cx,cy,tx,ty,w.color,d2*(1400/3),w.width);
  }});
}}
function qbp(t,x0,y0,cx,cy,x1,y1){{const m=1-t;return{{x:m*m*x0+2*m*t*cx+t*t*x1,y:m*m*y0+2*m*t*cy+t*t*y1}}}}
function animateDot(svg,fx,fy,cx,cy,tx,ty,color,delay,size){{
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('r',Math.max(3,size/1.5));c.setAttribute('fill',color);
  c.setAttribute('opacity','0');c.setAttribute('filter','url(#dotBlur)');
  svg.appendChild(c);
  let st=null;const dur=1400;
  function frame(ts){{
    if(!st)st=ts+delay;
    const el=ts-st;if(el<0){{requestAnimationFrame(frame);return}}
    const t=(el%dur)/dur;
    const pos=qbp(t,fx,fy,cx,cy,tx,ty);
    c.setAttribute('cx',pos.x);c.setAttribute('cy',pos.y);
    const op=t<.1?t/.1:t>.85?(1-t)/.15:1;
    c.setAttribute('opacity',op*.95);
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}
window.addEventListener('load',()=>setTimeout(()=>drawWires(),80));
window.addEventListener('resize',()=>{{document.querySelectorAll('.wire-tag').forEach(t=>t.remove());drawWires()}});
</script></body></html>"""


def show():
    st.title("⚡ Perceptron")
    st.markdown("""
    A **Perceptron** is the **simplest brain cell** in a neural network. 🧠

    It receives signals, **adds them up with weights**, and decides: **fire (1) or stay silent (0)**!

    `z = w1×x1 + w2×x2 + bias` → then `step(z)` decides the output
    """)

    st.divider()

    st.subheader("🔧 Logic Gate Demo")
    gate = st.selectbox("Choose a logic gate:", ["AND", "OR", "NAND", "NOR"])

    gate_data = {
        "AND":  {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [0,0,0,1]},
        "OR":   {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [0,1,1,1]},
        "NAND": {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [1,1,1,0]},
        "NOR":  {"inputs": [[0,0],[0,1],[1,0],[1,1]], "targets": [1,0,0,0]},
    }
    data    = gate_data[gate]
    inputs  = np.array(data["inputs"])
    targets = np.array(data["targets"])

    df = pd.DataFrame(inputs, columns=["x1", "x2"])
    df["Expected Output"] = targets
    st.markdown(f"**Truth table for {gate} gate:**")
    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("🎛️ Tune Weights & Watch the Brain React!")

    col1, col2, col3 = st.columns(3)
    with col1:
        w1   = st.slider("Weight 1 (w1)", -2.0, 2.0, 1.0, 0.1)
    with col2:
        w2   = st.slider("Weight 2 (w2)", -2.0, 2.0, 1.0, 0.1)
    with col3:
        bias = st.slider("Bias", -2.0, 2.0, -1.0, 0.1)

    st.markdown("**👇 Pick an input row — the diagram updates live:**")
    row_labels = [f"x1={int(r[0])}, x2={int(r[1])}" for r in inputs]
    sel     = st.radio("Input row", row_labels, horizontal=True)
    sel_idx = row_labels.index(sel)
    sx1, sx2 = int(inputs[sel_idx][0]), int(inputs[sel_idx][1])

    sz   = w1 * sx1 + w2 * sx2 + bias
    sout = step(sz)

    components.html(
        animated_perceptron_html(sx1, sx2, w1, w2, bias, sz, sout),
        height=545, scrolling=False
    )

    with st.expander("🔍 Show me the maths step by step", expanded=True):
        correct = (sout == int(targets[sel_idx]))
        st.markdown(f"""
| Step | What happens | Result |
|------|-------------|--------|
| 1️⃣ Multiply | `{w1:+.1f} × {sx1}` = `{w1*sx1:+.3f}` and `{w2:+.1f} × {sx2}` = `{w2*sx2:+.3f}` | weighted inputs |
| 2️⃣ Add up | `{w1*sx1:+.3f}` + `{w2*sx2:+.3f}` + `{bias:+.1f}` | **z = {sz:+.3f}** |
| 3️⃣ Decide | Is z (`{sz:+.3f}`) ≥ 0? → **{"YES 🟢" if sz>=0 else "NO 🔴"}** | **output = {sout}** |
| 4️⃣ Expected | — | **{int(targets[sel_idx])}** |
| 5️⃣ Correct? | — | {"✅ YES!" if correct else "❌ Nope, keep tuning!"} |
        """)

    st.divider()

    predictions = [step(w1*x[0] + w2*x[1] + bias) for x in inputs]
    df["Your Prediction"] = predictions
    df["✅ Correct?"]     = df["Expected Output"] == df["Your Prediction"]
    accuracy = df["✅ Correct?"].mean() * 100

    st.markdown(f"### Your accuracy: **{accuracy:.0f}%**")
    st.dataframe(df, use_container_width=True)
    if accuracy == 100:
        st.success("🎉 Amazing! You cracked the gate!")
    else:
        st.warning("❌ Not perfect yet — tweak the sliders above!")

    st.divider()

    st.subheader("🤖 Let the Computer Learn by Itself!")
    st.markdown("Watch how the perceptron **automatically finds the right weights** by trial and error!")

    lr     = st.slider("Learning Rate (how fast it learns)", 0.01, 1.0, 0.1, 0.01)
    epochs = st.slider("How many tries?", 10, 500, 100, 10)

    if st.button("🚀 Train Now!", type="primary"):
        w = np.zeros(2)
        b = 0.0
        loss_history = []

        for _ in range(epochs):
            total_error = 0
            for x, y in zip(inputs, targets):
                z    = np.dot(w, x) + b
                pred = step(z)
                err  = y - pred
                w   += lr * err * x
                b   += lr * err
                total_error += abs(err)
            loss_history.append(total_error)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=loss_history, mode="lines+markers", name="Mistakes",
            line=dict(color="#ff5252", width=3),
            marker=dict(size=4, color="#ff9800")
        ))
        fig_loss.update_layout(
            title="📉 Mistakes going down as it learns!",
            xaxis_title="Try number (epoch)", yaxis_title="Number of mistakes",
            template="plotly_dark", height=280
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("#### 🧠 Neuron After Training:")
        ex   = inputs[sel_idx]
        fz   = w[0]*ex[0] + w[1]*ex[1] + b
        fout = step(fz)
        components.html(
            animated_perceptron_html(int(ex[0]), int(ex[1]), w[0], w[1], b, fz, fout),
            height=545, scrolling=False
        )

        final_preds = [step(np.dot(w, x) + b) for x in inputs]
        df["Trained Prediction"] = final_preds
        acc = (np.array(final_preds) == targets).mean() * 100
        st.markdown(f"### Final Accuracy: **{acc:.0f}%**")
        st.dataframe(df[["x1","x2","Expected Output","Trained Prediction"]], use_container_width=True)
        st.success(f"🏆 Learned weights: w1={w[0]:.3f},  w2={w[1]:.3f},  bias={b:.3f}")

    st.divider()
    st.info("⚠️ A perceptron can only solve problems where you can draw one straight line to separate the answers. It cannot learn XOR!")