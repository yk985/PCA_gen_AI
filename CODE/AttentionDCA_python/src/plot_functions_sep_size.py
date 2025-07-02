import torch
import torch.nn.functional as F
from CODE.AttentionDCA_python.src.dcascore import correct_APC, compute_residue_pair_dist
import matplotlib.pyplot as plt
import numpy as np

def assemble_full_attention_map(model, make_inter_sym=True):
    """
    Reconstruct a full (H, N, N) attention map from a sub-block-based model.
    Now includes symmetrical bottom-left block for inter-domain 
    and correct diagonal mask for domain2.
    """
    device = model.Q.device
    H = model.H
    N = model.N

    L1 = model.N_alpha
    L2 = model.N_beta
    domain2_start = model.domain2_start

    A = torch.zeros(H, N, N, device=device)

    # 1) Domain1 heads
    for h in range(0, model.H1):
        e_sel = compute_e_sel_subblock(model, h, domain='domain1')
        # e_sel: (L1,L1)
        sf = F.softmax(e_sel, dim=1)
        A[h, 0:L1, 0:L1] = sf

    # 2) Domain2 heads
    for h in range(model.H1, model.H2):
        e_sel = compute_e_sel_subblock(model, h, domain='domain2')
        # e_sel: (L2,L2)
        sf = F.softmax(e_sel, dim=1)
        A[h, domain2_start:, domain2_start:] = sf

    # 3) Inter-domain heads (domain1->domain2)
    for h in range(model.H2, H):
        e_sel = compute_e_sel_subblock(model, h, domain='inter')
        # e_sel: (L1,L2)
        sf = F.softmax(e_sel, dim=1)
        A[h, 0:L1, domain2_start:] = sf

        # If we want symmetrical domain2->domain1 in the plot:
        if make_inter_sym:
            # The simplest approach is to transpose sf
            A[h, domain2_start:, 0:L1] = sf.transpose(0,1)

    # 4) Zero out diagonal for domain1 block
    for h in range(0, model.H1):
        A[h, 0:L1, 0:L1].fill_diagonal_(0)

    # 5) Zero out diagonal for domain2 block
    for h in range(model.H1, model.H2):
        A[h, domain2_start:, domain2_start:].fill_diagonal_(0)

    return A



def compute_e_sel_subblock(model, head_idx, domain='domain1'):
    """
    Compute the raw attention logits e_sel for a single head in the given domain block.
    domain ∈ {'domain1', 'domain2', 'inter'}

    Returns: e_sel (logits) as a 2D matrix
      - domain1 => shape (L1, L1)
      - domain2 => shape (L2, L2)
      - inter   => shape (L1, L2)
    """
    Q = model.Q[head_idx]  # shape (d, N)
    K = model.K[head_idx]  # shape (d, N)

    L1 = model.N_alpha
    L2 = model.N_beta
    start2 = model.domain2_start

    if domain == 'domain1':
        # Q_sel, K_sel => shape (d, L1)
        Q_sel = Q[:, :L1]
        K_sel = K[:, :L1]
        # e_sel => (L1, L1)
        e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)

    elif domain == 'domain2':
        Q_sel = Q[:, start2:]  # (d, L2)
        K_sel = K[:, start2:]  # (d, L2)
        # e_sel => (L2, L2)
        e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)

    elif domain == 'inter':
        # domain1 -> domain2
        Q_sel = Q[:, :L1]       # shape (d, L1)
        K_sel = K[:, start2:]   # shape (d, L2)
        e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)  # (L1, L2)

    else:
        raise ValueError("domain must be one of ['domain1','domain2','inter'].")

    # Optionally, if you want to exclude self-interactions even within domain blocks,
    # you can zero out the diagonal for domain1/domain2.
    # For domain='inter', there's no diagonal in a L1xL2 matrix anyway.

    return e_sel

def k_matrix_precomputed(A, k, version='mean', sym=True, APC=False, sqr=False):
    """
    Same logic as k_matrix, but we assume `A` is already shape (H, N, N)
    i.e. the full attention map across all heads.
    """
    device = A.device
    H, N, _ = A.shape
    if sqr:
        A = A * A  # Element-wise square
    
    if k >= N * (N - 1) / 2:
        if version == 'mean':
            M = torch.mean(A, dim=0)  # Shape: (N, N)
        elif version == 'maximum':
            M, _ = torch.max(A, dim=0)  # Shape: (N, N)
        
        M = (M + M.transpose(0, 1)) / 2  # Symmetrize
        
        if APC:
            M = correct_APC(M)
        
        return M, A
    
    _A = torch.zeros(H, N, N, device=A.device, dtype=A.dtype)
    
    for h in range(H):
        # Flatten the h-th attention matrix
        A_h = A[h].flatten()  # Shape: (N*N,)
        
        # Get top-k values and their indices
        vmins, idxs = torch.topk(A_h, k, largest=True, sorted=False)  # Shape: (k,)
        
        # Convert flat indices to 2D indices
        i_indices = idxs // N
        j_indices = idxs % N
        
        # Assign the top-k values to _A
        _A[h, i_indices, j_indices] = vmins
    
    if version == 'maximum':
        M, _ = torch.max(_A, dim=0)  # Shape: (N, N)
    elif version == 'mean':
        sum_A = _A.sum(dim=0)  # Shape: (N, N)
        count_A = (_A != 0).sum(dim=0)  # Shape: (N, N)
        M = torch.zeros(N, N, device=A.device, dtype=A.dtype)
        mask = count_A > 0
        M[mask] = sum_A[mask] / count_A[mask]
        
    
    M = (M + M.transpose(0, 1)) / 2  # Symmetrize
    
    if APC:
        M = correct_APC(M)
    
    return M, _A
def true_structure(structfile, min_separation=0, cutoff=8.0):
    """
    Compute the true structure coordinates based on residue pair distances.

    Parameters:
    - structfile: Path to the structure file
    - min_separation: Minimum separation between residue pairs
    - cutoff: Maximum allowed distance

    Returns:
    - coords: NumPy array of shape (2, M) where M is the number of residue pairs
    """
    # Compute residue pair distances
    # Assuming AttentionDCA.compute_residue_pair_dist returns a dictionary
    # where keys are tuples (i, j) and values are distances
    dist_dict = compute_residue_pair_dist(structfile)  # dict with keys=(i,j), values=distance
    
    # Filter the dictionary based on the criteria
    filtered_dist = {
        k: v for k, v in dist_dict.items()
        if (k[1] - k[0] > min_separation) and (v <= cutoff) and (v != 0)
    }
    
    # Extract the keys (pairs) and convert to a NumPy array
    # Each key is a tuple (i, j), so we create a 2xM array
    if not filtered_dist:
        return np.array([[], []])  # Return an empty array if no pairs meet the criteria
    
    pair_indices = list(filtered_dist.keys())  # List of tuples [(i1, j1), (i2, j2), ...]
    coords = np.array(pair_indices).T  # Shape: (2, M)
    
    return coords

def graphAtt_precomputed(M, A , filestruct, PFname, ticks, k=None, version='mean', sqr=False, APC=True, all = False):
    """
    Visualizes the Attention Map based on Q, K, V matrices.

    Parameters:
    - Q, K, V: PyTorch tensors with shapes (H, D, N)
    - filestruct: Structure data compatible with true_structure function
    - PFname: String for the plot title
    - ticks: List or array for axis ticks
    - k: Optional parameter for k_matrix function
    - version: Aggregation method ('mean' or 'maximum')
    - sqr: Boolean flag to square the attention matrix
    - APC: Boolean flag for APC correction
    """
    N = A.shape[1]
    ms = 100 / N
    # Replace diagonal elements with the mean of M
    mean_M = torch.mean(M)
    for i in range(M.size(0)):
        M[i, i] = mean_M
    
    # Convert M to NumPy for plotting
    Am = M.cpu().detach().numpy()

    if all:
        for i in range(len(A)):
            # Replace diagonal elements with the mean of M
            mean_M = torch.mean(A[i])
            for j in range(A[i].size(0)):
                A[i, j,j] = mean_M
            print(i+1)
            # Convert M to NumPy for plotting
            Am = A[i].cpu().detach().numpy()
    
            # Display the Attention Map
            plt.imshow(Am, cmap='gray_r', aspect='auto')
    
    
            # Retrieve and adjust true structure coordinates
            dist = true_structure(filestruct, min_separation=0, cutoff=8.0)  # Shape: (2, M)
    
            if dist.size != 0:
                # Scatter plot for True Structure
                plt.scatter((dist[0] - 1)*1, (dist[1] - 1)*1, c='r', marker='o', alpha=0.1, s=ms, label='True Structure')
                plt.scatter((dist[1] - 1)*1, (dist[0] - 1)*1, c='r', marker='o', alpha=0.1, s=ms)
    
            # Set axis ticks and labels
            plt.xticks(ticks - 1, ticks, fontsize=5)
            plt.yticks(ticks - 1, ticks, fontsize=5)
    
            # Set plot title
            plt.title(f"{PFname}, Attention Map", fontsize=20)
    
            # Add colorbar with specified shrink and aspect
            plt.colorbar(shrink=0.7, aspect=10)
    
            # Add legend
            plt.legend()
    
            # Display the plot
            plt.show()
            #save the image
             #plt.savefig(f"AttentionMap_{PFname}_testa{i+1}.pdf")

####################MEAN####################

    # Replace diagonal elements with the mean of M
    mean_M = torch.mean(M)
    for i in range(M.size(0)):
        M[i, i] = mean_M
    
    # Convert M to NumPy for plotting
    Am = M.cpu().detach().numpy()
    #print(Am)
 
    # Display the Attention Map
    plt.imshow(Am, cmap='gray_r')#, aspect='auto')
    plt.colorbar()#(shrink=0.7, aspect=10)
    # Retrieve and adjust true structure coordinates
    dist = true_structure(filestruct, min_separation=0, cutoff=8.0)  # Shape: (2, M)
    if dist.size != 0:
        # Scatter plot for True Structure
        plt.scatter((dist[0] - 1)*1, (dist[1] - 1)*1, c='r', marker='o', alpha=0.1, s=ms, label='True Structure')
        plt.scatter((dist[1] - 1)*1, (dist[0] - 1)*1, c='r', marker='o', alpha=0.1, s=ms)
    # Set axis ticks and labels
    plt.xticks(ticks - 1, ticks, fontsize=5)
    plt.yticks(ticks - 1, ticks, fontsize=5)
    # Set plot title
    plt.title(f"{PFname}, Attention Map", fontsize=20)
    #save the image
    #plt.savefig(f"AttentionMap_{PFname}_mean.pdf")
    # Add colorbar with specified shrink and aspect
    
    # Add legend
    plt.legend()
    