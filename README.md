# 🧠 Neural Network Toolbox (Simple Version)

A lightweight, easy-to-run Streamlit app to learn Neural Network concepts interactively.

## 📦 Modules Included

| Module | What you learn |
|---|---|
| ⚡ Perceptron | Logic gates, weights, bias, manual tuning + auto-train |
| ➡️ Forward Propagation | Step-by-step data flow through layers |
| ⬅️ Backward Propagation | How gradients flow, weight updates |
| 🔗 MLP Classifier | Full neural network on real datasets (Iris, Cancer, or your CSV) |

---

## 🚀 How to Run

### Step 1: Make sure Python is installed
Open your terminal / command prompt and run:
```
python --version
```
You need Python 3.8 or higher.

---

### Step 2: Install dependencies

Navigate to this folder, then run:
```
pip install -r requirements.txt
```

---

### Step 3: Run the app
```
streamlit run app.py
```

The app will open automatically in your browser at: `http://localhost:8501`

---

## 📁 Folder Structure

```
nn_toolbox/
├── app.py                    ← Main entry point
├── requirements.txt          ← Python packages
├── README.md
└── src/
    └── pages/
        ├── home.py           ← Home page
        ├── perceptron.py     ← Perceptron demo
        ├── forward_prop.py   ← Forward propagation
        ├── backward_prop.py  ← Backpropagation
        └── mlp.py            ← MLP classifier
```

---

## 💡 No GPU needed — everything runs on CPU!
