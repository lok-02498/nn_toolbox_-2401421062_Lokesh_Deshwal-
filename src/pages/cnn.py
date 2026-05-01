import streamlit as st
import math


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def convolve(image, kernel):
    output = []

    for i in range(len(image) - 2):
        row = []
        for j in range(len(image[0]) - 2):
            value = 0
            for ki in range(3):
                for kj in range(3):
                    value += image[i + ki][j + kj] * kernel[ki][kj]
            row.append(value)
        output.append(row)

    return output


def apply_relu(matrix):
    return [[relu(value) for value in row] for row in matrix]


def max_pooling(matrix):
    pooled = []

    for i in range(0, len(matrix), 2):
        row = []
        for j in range(0, len(matrix[0]), 2):
            block = []

            for bi in range(2):
                for bj in range(2):
                    if i + bi < len(matrix) and j + bj < len(matrix[0]):
                        block.append(matrix[i + bi][j + bj])

            row.append(max(block))
        pooled.append(row)

    return pooled


def flatten(matrix):
    return [value for row in matrix for value in row]


def dense_predict(flat_values, weights, bias):
    total = bias

    for x, w in zip(flat_values, weights):
        total += x * w

    probability = sigmoid(total)
    return probability, total


def show():
    st.title("🧩 CNN ")
    st.markdown("### Beginner-friendly CNN with Forward Pass and Backward Pass")

    st.info("This model uses only Python + Streamlit. No PyTorch, TensorFlow, NumPy, or OpenCV.")

    st.divider()

    st.subheader("1️⃣ What is CNN?")

    st.markdown("""
    A **CNN**, or **Convolutional Neural Network**, is mainly used for image data.

    CNN learns patterns such as:

    - Edges
    - Lines
    - Shapes
    - Textures
    - Objects

    In this demo, the image is represented as a small matrix.
    """)

    st.divider()

    st.subheader("2️⃣ Input Image Matrix")

    image = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]

    st.table(image)

    st.markdown("""
    Here:

    - `1` means bright pixel
    - `0` means dark pixel
    """)

    st.divider()

    st.subheader("3️⃣ Kernel / Filter")

    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ]

    st.table(kernel)

    st.markdown("""
    This is a **vertical edge detector**.

    The kernel slides over the image and finds patterns.
    """)

    st.divider()

    st.subheader("4️⃣ Convolution Operation")

    st.latex(r"Feature\ Map = Image * Kernel")

    st.markdown("""
    Convolution means:

    1. Place kernel on a small part of image.
    2. Multiply image values with kernel values.
    3. Add all values.
    4. Move kernel to next position.
    """)

    feature_map = convolve(image, kernel)

    st.write("Feature Map after Convolution:")
    st.table(feature_map)

    st.divider()

    st.subheader("5️⃣ ReLU Activation")

    st.latex(r"ReLU(x)=max(0,x)")

    st.markdown("""
    ReLU removes negative values.

    Negative values are replaced with `0`.
    """)

    relu_map = apply_relu(feature_map)

    st.write("After ReLU:")
    st.table(relu_map)

    st.divider()

    st.subheader("6️⃣ Max Pooling")

    st.markdown("""
    Max pooling reduces the size of the feature map.

    It keeps the strongest feature from each small block.
    """)

    pooled_map = max_pooling(relu_map)

    st.write("After Max Pooling:")
    st.table(pooled_map)

    st.divider()

    st.subheader("7️⃣ Flatten Layer")

    flat_values = flatten(pooled_map)

    st.markdown("""
    Flatten converts the matrix into a simple list so it can go into the dense layer.
    """)

    st.write(flat_values)

    st.divider()

    st.subheader("8️⃣ Dense Layer Prediction")

    dense_weights = [0.4 for _ in flat_values]

    bias = st.slider("Bias", -2.0, 2.0, 0.0, 0.1)

    probability, raw_score = dense_predict(flat_values, dense_weights, bias)

    st.latex(r"z = w_1x_1 + w_2x_2 + ... + b")
    st.latex(r"prediction = sigmoid(z)")

    col1, col2 = st.columns(2)

    col1.metric("Raw Score", round(raw_score, 4))
    col2.metric("Probability", round(probability, 4))

    if probability >= 0.5:
        prediction = 1
        st.success("Prediction: Object Detected")
    else:
        prediction = 0
        st.error("Prediction: No Object")

    st.divider()

    st.subheader("9️⃣ Loss Calculation")

    actual = st.selectbox("Actual Label", [1, 0])

    loss = (probability - actual) ** 2

    st.latex(r"Loss = (Predicted - Actual)^2")

    st.metric("Loss", round(loss, 4))

    st.divider()

    st.subheader("🔟 Backward Pass")

    st.markdown("""
    Backward pass means the model learns from error.

    If prediction is wrong, weights are changed.

    Simple weight update formula:
    """)

    st.latex(r"new\ weight = old\ weight - learning\ rate \times gradient")

    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)

    error = probability - actual

    gradients = [error * x for x in flat_values]

    updated_weights = []

    for old_weight, gradient in zip(dense_weights, gradients):
        new_weight = old_weight - learning_rate * gradient
        updated_weights.append(new_weight)

    st.write("Old Weights:")
    st.write([round(w, 4) for w in dense_weights])

    st.write("Gradients:")
    st.write([round(g, 4) for g in gradients])

    st.write("Updated Weights:")
    st.write([round(w, 4) for w in updated_weights])

    st.divider()

    st.subheader("🔁 Prediction After Learning")

    new_probability, new_raw_score = dense_predict(flat_values, updated_weights, bias)
    new_loss = (new_probability - actual) ** 2

    col1, col2, col3 = st.columns(3)

    col1.metric("Old Probability", round(probability, 4))
    col2.metric("New Probability", round(new_probability, 4))
    col3.metric("New Loss", round(new_loss, 4))

    if new_probability >= 0.5:
        st.success("New Prediction: Object Detected")
    else:
        st.error("New Prediction: No Object")

    st.divider()

    st.subheader("✅ Final Summary")

    st.markdown("""
    CNN working flow:

    1. Image is taken as matrix.
    2. Kernel scans the image.
    3. Convolution creates feature map.
    4. ReLU removes negative values.
    5. Pooling reduces size.
    6. Flatten converts matrix into list.
    7. Dense layer makes prediction.
    8. Loss checks error.
    9. Backward pass updates weights.
    """)