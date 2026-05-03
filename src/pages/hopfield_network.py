import numpy as np

def calculate_weight_matrix(patterns, method="hebbian"):
    patterns = np.array(patterns)

    if method == "pseudoinverse":
        W = patterns.T @ np.linalg.pinv(patterns.T)
    else:
        W = patterns.T @ patterns / patterns.shape[0]

    np.fill_diagonal(W, 0)
    return W


def apply_noise(pattern, noise_level):
    noisy = pattern.copy()
    n_flip = int(len(noisy) * noise_level)

    if n_flip > 0:
        flip_indices = np.random.choice(len(noisy), n_flip, replace=False)
        noisy[flip_indices] *= -1

    return noisy


def async_recall(state, W, max_epochs=30):
    state = state.copy()
    n = len(state)

    for _ in range(max_epochs):
        old_state = state.copy()

        for i in np.random.permutation(n):
            activation = np.dot(W[i], state)
            state[i] = 1 if activation >= 0 else -1

        if np.array_equal(state, old_state):
            break

    return state


def calculate_energy(state, W):
    return -0.5 * state.T @ W @ state