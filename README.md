# 🧠 Neural Network Toolbox (Advanced Interactive Version)

An interactive **Streamlit-based learning platform** to understand Neural Networks from **basics → advanced concepts** with **live demos, animations, and real-world mini projects**.

---

## 📦 Modules Included

| Module | What you learn |
|---|---|
| ⚡ **Perceptron** | Logic gates, weights, bias, training |
| ➡️ **Forward Propagation** | Animated data flow through layers |
| ⬅️ **Backward Propagation** | Gradients, loss, weight updates |
| 🔗 **MLP Classifier** | Multi-layer neural network on datasets |
| 📈 **RNN (From Scratch)** | Sequence learning & next-word prediction |
| 🧩 **CNN (From Scratch)** | Image feature extraction basics |
| 💬 **Sentiment Analysis** | Real ML model using TF-IDF + Decision Tree |
| 📷 **Face Detection** | OpenCV-based face detection system |
| 🧠 **Hopfield OCR** | Pattern recognition using Hopfield Networks |

---

## 🚀 How to Run

### Step 1: Install Python

Check Python version:

python --version


Requires Python 3.8+

---

### Step 2: Install dependencies

Navigate to project folder:

pip install -r requirements.txt


---

### Step 3: Run the app


streamlit run app.py


Open in browser:

http://localhost:8501


---

## 📁 Project Structure


nn_toolbox/
├── app.py ← Main app (routing)
├── requirements.txt ← Dependencies
├── README.md
├── sentiment_model.pkl ← Trained ML model
├── flipkardata.xlsx ← Dataset (optional)
├── registered_students.csv ← Face registration data

└── src/
├── pages/
│ ├── home.py
│ ├── perceptron.py
│ ├── forward_prop.py
│ ├── backward_prop.py
│ ├── mlp.py
│ ├── rnn.py
│ ├── cnn.py
│ ├── sentiment.py
│ ├── face_detection.py
│ ├── hopfield_ocr.py
│ └── hopfield_network.py

└── preprocessing.py          ← Text preprocessing

---

## 🧠 Key Features

### 🎯 Beginner Friendly
- Step-by-step explanations
- Interactive sliders
- Visual outputs

### 🎨 Advanced UI
- Animated neural networks
- Live graph updates
- Clean and modern interface

### 🔬 From Scratch Implementations
- RNN (without heavy libraries)
- CNN basics
- Hopfield Network

### 🤖 Real-world Integrations
- Sentiment Analysis (ML model)
- Face Detection (OpenCV)
- OCR using Hopfield Network

---

## ⚠️ Important Notes

### PyTorch Issue (DLL Error)

If you faced:

DLL initialization failed (torch)


You replaced RNN with **from-scratch implementation**, so no heavy dependency is required.

---

### OpenCV Installation

If error occurs:

pip install opencv-python-headless


---

### Canvas Feature

For Hopfield OCR:

pip install streamlit-drawable-canvas


---

## 🎯 Learning Flow Recommendation


Perceptron → Forward Propagation → Backward Propagation → MLP → RNN → CNN → Hopfield OCR → Real Projects


---

## 💡 No GPU Required

Runs fully on CPU and is lightweight.

---

## 🚀 Future Improvements

- Add MNIST digit classifier  
- Add LSTM / GRU demo  
- Improve Hopfield accuracy  
- Add training visualizations  

---

## 👨‍💻 Author

Developed by **Lokesh Deshwal**

---

## ⭐ Final Note

This project is a **complete Neural Network learning toolkit + mini AI lab** combining:
