# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
import os
cwd = os.getcwd()
sys.path.insert(1, cwd + '/CODE/AttentionDCA_python/src')
import model
from model import AttentionModel, loss_wo_J


import torch
import numpy as np


def _loss(Q, K, V, Z, W, lambd=0.001):
    H, d, L = Q.shape
    _, _, _ = K.shape
    _, q, _ = V.shape
    L, M = Z.shape

    # Initialize sf tensor
    sf = torch.zeros(H, L, L, dtype=torch.float64)

    # Compute sf[h, i, j]
    for h in range(H):
        for i in range(L):
            for j in range(L):
                for n in range(d):
                    sf[h, i, j] += Q[h, n, i] * K[h, n, j]

    # Apply softmax over dimension j (axis=2)
    sf = torch.softmax(sf, dim=2)  # Shape: (H, L, L)

    # Initialize J tensor
    J = torch.zeros(L, L, q, q, dtype=torch.float64)

    # Compute J[i, j, a, b]
    for i in range(L):
        for j in range(L):
            if j != i:
                for h in range(H):
                    for a in range(q):
                        for b in range(q):
                            J[i, j, a, b] += sf[h, i, j] * V[h, a, b]

    # Initialize mat_ene tensor
    mat_ene = torch.zeros(q, L, M, dtype=torch.float64)

    # Compute mat_ene[a, i, m]
    for m in range(M):
        for a in range(q):
            for i in range(L):
                for j in range(L):
                    mat_ene[a, i, m] += J[i, j, a, Z[j, m] - 1]  # Adjusting for zero-based indexing

    # Compute lge
    m = torch.max(mat_ene, dim=0, keepdim=True)[0]
    lge = m + torch.log(torch.sum(torch.exp(mat_ene - m), dim=0, keepdim=True))
    lge = lge[0, :, :]  # Shape: (L, M)

    # Compute pl
    pl = 0.0
    for i in range(L):
        for m_idx in range(M):
            z_im = Z[i, m_idx] - 1  # Adjusting for zero-based indexing
            pl -= W[m_idx] * (mat_ene[z_im, i, m_idx] - lge[i, m_idx])

    # Add regularization term
    pl += lambd * torch.sum(J ** 2)

    return pl


def test_loss():
    # Set a random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Small test data
    Z = torch.randint(1, 3, (2, 2), dtype=torch.long)  # Random integers between 1 and 2
    W = torch.full((2,), 1/2, dtype=torch.float64)     # Vector of weights
    Q = torch.rand(2, 2, 2, dtype=torch.float64)       # Random tensor of shape (H, d, N)
    K = torch.rand(2, 2, 2, dtype=torch.float64)
    V = torch.rand(2, 2, 2, dtype=torch.float64)

    # Compute the loss using the simplified loss function
    simplified_loss = _loss(Q, K, V, Z, W, lambd=0.001)


    # Adjust Z for zero-based indexing if necessary
    if Z.min().item() == 1:
      Z = Z - 1

    # Initialize the model with the test parameters
    model = AttentionModel(H=2, d=2, N=2, q=2)
    model.Q.data = Q
    model.K.data = K
    model.V.data = V

    # Compute the loss using the main loss function
    main_loss = loss_wo_J(Q, K, V, Z, W, lambd=0.001)


    # Compare the two losses
    assert torch.isclose(main_loss, simplified_loss, atol=1e-6), f"Losses do not match: {main_loss} vs {simplified_loss}"
    print("Test passed: The losses match.")


if __name__ == "__main__":
    test_loss()
