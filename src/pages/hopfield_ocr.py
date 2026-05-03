import streamlit as st
import numpy as np

from src.pages.hopfield_network import (
    calculate_weight_matrix,
    apply_noise,
    async_recall,
    calculate_energy,
)

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    CANVAS_AVAILABLE = False


GRID_SIZE = 15
NEURONS = GRID_SIZE * GRID_SIZE


LETTER_BITMAPS_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _bitmap_to_pattern(bitmap_rows):
    small = np.array(
        [[1 if ch == "1" else -1 for ch in row] for row in bitmap_rows],
        dtype=int
    )

    scaled = np.kron(small, np.ones((2, 2), dtype=int))

    grid = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)

    start_row = (GRID_SIZE - scaled.shape[0]) // 2
    start_col = (GRID_SIZE - scaled.shape[1]) // 2

    grid[
        start_row:start_row + scaled.shape[0],
        start_col:start_col + scaled.shape[1]
    ] = scaled

    return grid.flatten()


def _build_alphabet_patterns():
    labels = sorted(LETTER_BITMAPS_5X7.keys())
    patterns = np.array([
        _bitmap_to_pattern(LETTER_BITMAPS_5X7[label])
        for label in labels
    ])
    return labels, patterns


def _shift_bipolar_grid(grid, dr, dc):
    shifted = np.full_like(grid, -1)

    if dr >= 0:
        src_r = slice(0, GRID_SIZE - dr)
        dst_r = slice(dr, GRID_SIZE)
    else:
        src_r = slice(-dr, GRID_SIZE)
        dst_r = slice(0, GRID_SIZE + dr)

    if dc >= 0:
        src_c = slice(0, GRID_SIZE - dc)
        dst_c = slice(dc, GRID_SIZE)
    else:
        src_c = slice(-dc, GRID_SIZE)
        dst_c = slice(0, GRID_SIZE + dc)

    shifted[dst_r, dst_c] = grid[src_r, src_c]
    return shifted


def _dilate_bipolar_grid(grid):
    ink = grid == 1
    padded = np.pad(ink, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(ink)

    for r in range(3):
        for c in range(3):
            dilated |= padded[r:r + GRID_SIZE, c:c + GRID_SIZE]

    return np.where(dilated, 1, -1)


def _build_classifier_variants(base_patterns):
    variants = []
    variant_label_idx = []

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for label_idx, flat_pattern in enumerate(base_patterns):
        grid = flat_pattern.reshape(GRID_SIZE, GRID_SIZE)

        pool = [grid]
        pool.extend(_shift_bipolar_grid(grid, dr, dc) for dr, dc in shifts)

        thick = _dilate_bipolar_grid(grid)
        pool.append(thick)
        pool.extend(_shift_bipolar_grid(thick, dr, dc) for dr, dc in shifts)

        seen = set()

        for variant in pool:
            key = variant.tobytes()

            if key in seen:
                continue

            seen.add(key)
            variants.append(variant.flatten())
            variant_label_idx.append(label_idx)

    return np.array(variants, dtype=int), np.array(variant_label_idx, dtype=int)


def _aggregate_variant_scores(variant_scores, variant_to_label, num_labels):
    label_scores = np.full(num_labels, -np.inf, dtype=np.float64)

    for idx, score in enumerate(variant_scores):
        label_idx = int(variant_to_label[idx])

        if score > label_scores[label_idx]:
            label_scores[label_idx] = score

    return label_scores


def _center_ink(grid):
    ink = grid == 1

    if not np.any(ink):
        return grid

    rows = np.where(np.any(ink, axis=1))[0]
    cols = np.where(np.any(ink, axis=0))[0]

    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]

    glyph = grid[r0:r1 + 1, c0:c1 + 1]

    centered = np.full_like(grid, -1)

    gh, gw = glyph.shape
    start_row = (GRID_SIZE - gh) // 2
    start_col = (GRID_SIZE - gw) // 2

    centered[start_row:start_row + gh, start_col:start_col + gw] = glyph

    return centered


def _canvas_to_flat_bipolar(image_data):
    if image_data is None:
        return None

    gray = image_data[:, :, 0].astype(np.float32)

    pooled = gray.reshape(GRID_SIZE, 10, GRID_SIZE, 10).mean(axis=(1, 3))

    bipolar_grid = np.where(pooled < 220.0, 1, -1)

    centered = _center_ink(bipolar_grid)

    return centered.flatten()


def _shape_similarity_scores(state, patterns):
    s = np.where(np.asarray(state) >= 0, 1, -1).reshape(1, -1)
    p = np.where(np.asarray(patterns) >= 0, 1, -1)

    s_ink = s == 1
    p_ink = p == 1

    intersection = np.sum(p_ink & s_ink, axis=1).astype(np.float64)
    union = np.sum(p_ink | s_ink, axis=1).astype(np.float64)

    iou = intersection / np.maximum(union, 1.0)

    s_ink_count = float(np.sum(s_ink))
    precision = intersection / max(s_ink_count, 1.0)

    p_ink_count = np.sum(p_ink, axis=1).astype(np.float64)
    recall = intersection / np.maximum(p_ink_count, 1.0)

    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-9)

    dot = (p @ s.ravel()) / s.shape[1]
    dot01 = (dot + 1.0) / 2.0

    p_grid = p.reshape(p.shape[0], GRID_SIZE, GRID_SIZE)
    s_grid = s.reshape(GRID_SIZE, GRID_SIZE)

    p_ink_float = (p_grid == 1).astype(np.float64)
    s_ink_float = (s_grid == 1).astype(np.float64)

    p_row = np.sum(p_ink_float, axis=2)
    s_row = np.sum(s_ink_float, axis=1)

    row_dot = p_row @ s_row
    row_norm = np.linalg.norm(p_row, axis=1) * max(np.linalg.norm(s_row), 1e-9)
    row_sim = row_dot / np.maximum(row_norm, 1e-9)

    p_col = np.sum(p_ink_float, axis=1)
    s_col = np.sum(s_ink_float, axis=0)

    col_dot = p_col @ s_col
    col_norm = np.linalg.norm(p_col, axis=1) * max(np.linalg.norm(s_col), 1e-9)
    col_sim = col_dot / np.maximum(col_norm, 1e-9)

    return (
        0.35 * iou
        + 0.20 * f1
        + 0.10 * dot01
        + 0.20 * row_sim
        + 0.15 * col_sim
    )


def _classify_with_consensus(input_state, recalled_state):
    input_variant_scores = _shape_similarity_scores(input_state, classifier_patterns)
    recall_variant_scores = _shape_similarity_scores(recalled_state, classifier_patterns)

    input_scores = _aggregate_variant_scores(
        input_variant_scores,
        classifier_variant_to_label,
        len(ideal_labels),
    )

    recall_scores = _aggregate_variant_scores(
        recall_variant_scores,
        classifier_variant_to_label,
        len(ideal_labels),
    )

    agreement = float(np.mean(np.asarray(input_state) == np.asarray(recalled_state)))

    recall_weight = 0.15 + 0.25 * agreement
    input_weight = 1.0 - recall_weight

    combined_scores = (input_weight * input_scores) + (recall_weight * recall_scores)

    best_idx = int(np.argmax(combined_scores))

    sorted_scores = np.sort(combined_scores)

    if sorted_scores.size > 1:
        margin = float(sorted_scores[-1] - sorted_scores[-2])
    else:
        margin = float(sorted_scores[-1])

    return best_idx, combined_scores, input_scores, recall_scores, margin, agreement


ideal_labels, ideal_patterns = _build_alphabet_patterns()
W_matrix = calculate_weight_matrix(ideal_patterns, method="pseudoinverse")
classifier_patterns, classifier_variant_to_label = _build_classifier_variants(ideal_patterns)


def show():
    st.title("🧠 Hopfield Network OCR")
    st.markdown("""
    Draw a letter from **A to Z**.  
    The Hopfield Network will try to reconstruct and recognize the letter.
    """)

    if not CANVAS_AVAILABLE:
        st.warning("streamlit-drawable-canvas is not installed.")
        st.code("pip install streamlit-drawable-canvas")
        return

    if "canvas_result" not in st.session_state:
        st.session_state.canvas_result = None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1️⃣ Draw Letter")
        st.caption("Draw one uppercase letter")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=150,
            height=150,
            drawing_mode="freedraw",
            key="hopfield_canvas",
        )

        if canvas_result.image_data is not None:
            st.session_state.canvas_result = canvas_result

    with col2:
        st.subheader("2️⃣ Add Noise")

        noise_level = st.slider(
            "Noise %",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=5.0
        )

        recall_steps = st.slider(
            "Recall Steps",
            min_value=5,
            max_value=120,
            value=30,
            step=5
        )

        recall_attempts = st.slider(
            "Recall Attempts",
            min_value=1,
            max_value=12,
            value=5,
            step=1
        )

        if (
            st.session_state.canvas_result is not None
            and st.session_state.canvas_result.image_data is not None
        ):
            flat_input = _canvas_to_flat_bipolar(
                st.session_state.canvas_result.image_data
            )

            noisy_state = apply_noise(flat_input, noise_level / 100.0)

            display_noisy_grid = np.where(
                noisy_state.reshape(GRID_SIZE, GRID_SIZE) == 1,
                0,
                255
            ).astype(np.uint8)

            display_img = np.kron(
                display_noisy_grid,
                np.ones((10, 10))
            ).astype(np.uint8)

            st.image(display_img, caption=f"With {int(noise_level)}% noise", width=150)

    with col3:
        st.subheader("3️⃣ Recall Output")

        if st.button("🔄 Run Hopfield Recall"):
            if st.session_state.canvas_result is None:
                st.error("Please draw a letter first.")
                return

            flat_input = _canvas_to_flat_bipolar(
                st.session_state.canvas_result.image_data
            )

            if flat_input is None or np.count_nonzero(flat_input == 1) < 3:
                st.error("No clear drawing detected. Draw a bigger letter.")
                return

            noisy_input = apply_noise(flat_input, noise_level / 100.0)

            trials = []

            for _ in range(recall_attempts):
                candidate = async_recall(
                    noisy_input,
                    W_matrix,
                    max_epochs=recall_steps
                )

                (
                    best_idx,
                    combined_scores,
                    input_scores,
                    recall_scores,
                    margin,
                    agreement
                ) = _classify_with_consensus(noisy_input, candidate)

                confidence = float(combined_scores[best_idx])
                energy = calculate_energy(candidate, W_matrix)

                trials.append(
                    (
                        confidence,
                        margin,
                        agreement,
                        -energy,
                        candidate,
                        best_idx,
                        combined_scores,
                        input_scores,
                        recall_scores,
                    )
                )

            (
                _,
                best_margin,
                best_agreement,
                _,
                stable_state,
                best_match_idx,
                combined_similarities,
                input_similarities,
                recall_similarities,
            ) = max(trials, key=lambda x: (x[0], x[1], x[2], x[3]))

            display_stable_grid = np.where(
                stable_state.reshape(GRID_SIZE, GRID_SIZE) == 1,
                0,
                255
            ).astype(np.uint8)

            display_stable_img = np.kron(
                display_stable_grid,
                np.ones((10, 10))
            ).astype(np.uint8)

            st.image(display_stable_img, caption="Reconstructed", width=150)

            top_confidence = float(combined_similarities[best_match_idx])

            st.success(
                f"Prediction: {ideal_labels[best_match_idx]} "
                f"({top_confidence * 100:.1f}%)"
            )

            if top_confidence < 0.62 or best_margin < 0.03:
                top3_idx = np.argsort(combined_similarities)[-3:][::-1]

                top3_text = ", ".join(
                    f"{ideal_labels[i]} ({combined_similarities[i] * 100:.1f}%)"
                    for i in top3_idx
                )

                st.warning("Low confidence. Draw more clearly or increase recall steps.")
                st.caption(f"Top candidates: {top3_text}")
                st.caption(f"Input-recall agreement: {best_agreement * 100:.1f}%")

            with st.expander("📊 Similarity Scores"):
                for label, c_sim, i_sim, r_sim in zip(
                    ideal_labels,
                    combined_similarities,
                    input_similarities,
                    recall_similarities,
                ):
                    st.write(
                        f"{label}: combined {c_sim * 100:.1f}% | "
                        f"input {i_sim * 100:.1f}% | "
                        f"recall {r_sim * 100:.1f}%"
                    )

    st.divider()

    with st.expander("ℹ️ How Hopfield Network Works"):
        st.markdown("""
        A Hopfield Network is an **associative memory network**.

        It stores patterns and can recover them even when input is noisy.

        Steps:

        1. Store ideal A-Z letter patterns.
        2. User draws a letter.
        3. Noise is optionally added.
        4. Network updates neurons repeatedly.
        5. It converges to the closest stored pattern.
        6. The closest letter is predicted.

        Main formula:

        ```text
        xᵢ(t+1) = sign(Σ Wᵢⱼ xⱼ)
        ```
        """)


if __name__ == "__main__":
    show()