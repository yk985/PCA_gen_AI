import numpy as np
from CODE.AttentionDCA_python.src.utils import compute_weights, compute_weights_large
N = 53

# Small dataset example
Z_small = np.random.randint(1, 21, size=(N, 1000))  # Adjust size as needed
theta = 0.2

W_vectorized, Meff_vectorized = compute_weights(Z_small, theta)
W_large, Meff_large = compute_weights_large(Z_small, theta)

# Compare results
assert np.allclose(W_vectorized, W_large), "Weights do not match!"
assert np.isclose(Meff_vectorized, Meff_large), "Meff values do not match!"
print("Verification passed: Outputs match.")