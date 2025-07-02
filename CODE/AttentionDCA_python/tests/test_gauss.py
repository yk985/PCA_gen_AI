import torch
import numpy as np
from CODE.AttentionDCA_python.src.dcascore import score
from CODE.AttentionDCA_python.src.utils import load_matrix_3d

import numpy as np

from julia import Julia

# Initialize Julia
jl = Julia(compiled_modules=False)  # Set compiled_modules=True if you face issues


from julia import Julia, PottsGauge, Main

# Import the necessary Julia packages
Main.using("PottsGauge")










def _gauge(Q, K, V):
    H, d, L = Q.shape
    _, _, _ = K.shape
    _, q, _ = V.shape
    device = Q.device

    N = L

    # Compute W[h, i, j]
    W = torch.einsum('hdi,hdj->hij', Q, K)  # Shape: (H, L, L)
    W = torch.softmax(W, dim=2)  # Softmax over j

    # Apply mask (i != j)
    i_indices = torch.arange(L, device=device).unsqueeze(1)
    j_indices = torch.arange(L, device=device).unsqueeze(0)
    mask = (i_indices != j_indices).float().unsqueeze(0)
    W = W * mask
    

    # Compute Jtens
    Jtens = torch.einsum('hri,hab->abri', W, V)  # Shape: (q, q, L, L)

    # Ensure Jtens is on CPU and detached from any computation graph
    Jtens_cpu = Jtens.cpu().detach()

    # Convert to NumPy array with dtype float64
    Jtens_np = Jtens_cpu.numpy().astype(np.float64)

    # Ensure Jtens is on CPU and detached from any computation graph
    Jtens_cpu = Jtens.cpu().detach()

    # Convert to NumPy array with dtype float64
    Jtens_np = Jtens_cpu.numpy().astype(np.float64)

    # Convert NumPy array to Julia Array{Float64,4}
    JArrayType = Main.eval("Array{Float64,4}")  # Obtain the Julia type
    
    Jtens_julia = Main.convert(JArrayType, Jtens_np)

    # Perform the permutedims operation in Julia
    Jt = 0.5 * (Jtens_julia + Main.permutedims(Jtens_julia, [2, 1, 4, 3]))

    # Create a Julia zeros array for ht with shape (q, N)
    ht = Main.zeros(Main.eltype(Jt), q, N)

    # Call the gauge function from PottsGauge
    Jzsg, _ = PottsGauge.gauge(Jt, ht, PottsGauge.ZeroSumGauge())

    # Convert Jzsg to NumPy array
    Jzsg_np = np.array(Jzsg)

    return Jzsg_np, _



def test_gauss():

    torch.manual_seed(0)
    np.random.seed(0)

    # Small test data
    Q_np = load_matrix_3d("./CODE/AttentionDCA.jl/new_Q_values_32H_23d.txt")
    K_np = load_matrix_3d("./CODE/AttentionDCA.jl/new_K_values_32H_23d.txt")
    V_np = load_matrix_3d("./CODE/AttentionDCA.jl/new_V_values_32H_22q.txt") 

    Q = torch.from_numpy(Q_np).float()  # Shape: (32, 23, 53)
    K = torch.from_numpy(K_np).float()  # Shape: (32, 23, 53)
    V = torch.from_numpy(V_np).float()  # Shape: (23, 22, 22)

    _, Jzsg_np = score(Q, K, V)

    JT_np, hT = _gauge(Q, K, V)

    assert np.allclose(Jzsg_np, JT_np)
    print("All tests passed!")

if __name__ == '__main__':
    test_gauss()




