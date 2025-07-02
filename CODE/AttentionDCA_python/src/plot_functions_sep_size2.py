import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from CODE.AttentionDCA_python.src.dcascore import correct_APC, compute_residue_pair_dist

###############################################################################
#  1) Build the full (H, N, N) attention map from a MultiDomainAttentionSubBlock
#     but only for the chosen sub-block: 'domain1', 'domain2', 'inter', or 'all'.
###############################################################################
import torch
import torch.nn.functional as F

def assemble_full_attention_map_subblock(model, subblock='all', make_inter_sym=False):
    """
    Reconstruct a full (H, N, N) attention map from a MultiDomainAttentionSubBlock model
    that has physically separate:
      - Q1, K1, V1 for domain1 heads => range: [0..H1)
      - Q2, K2, V2 for domain2 heads => range: [H1..H2)
      - Qint1,Kint1 for domain1->domain2 => shape: (num_inter_heads1,...)
      - Qint2,Kint2 for domain2->domain1 => shape: (num_inter_heads2,...)

    In the constructor, we have something like:
        num_inter_heads  = H - H2
        num_inter_heads1 = num_inter_heads // 2
        num_inter_heads2 = num_inter_heads - num_inter_heads1

    So the total # of inter heads is num_inter_heads1 + num_inter_heads2 = (H - H2).

    We want to fill these inter heads in the final attention map as follows:
      - For h in [H2 .. H2+num_inter_heads1): domain1->domain2
      - For h in [H2+num_inter_heads1 .. H): domain2->domain1

    If subblock='all', we fill domain1 + domain2 + both inter sub-blocks.
    If subblock='domain1', we only fill domain1 heads. 
    If subblock='domain2', we only fill domain2 heads.
    If subblock='inter',  we only fill inter heads (both domain1->domain2 and domain2->domain1).

    If make_inter_sym=True, we *also* mirror domain1->domain2 attention into the bottom-left,
    but typically you'll skip that if you have *real* domain2->domain1 parameters (Qint2,Kint2).
    """
    device = model.Q1.device
    H = model.H
    N = model.N

    L1 = model.N_alpha         # domain1 length
    L2 = model.N_beta          # domain2 length
    dstart = model.domain2_start

    # Some bookkeeping:
    num_inter_heads  = H - model.H2
    num_inter_heads1 = model.Qint1.shape[0]  # or num_inter_heads // 2
    num_inter_heads2 = model.Qint2.shape[0]  # or num_inter_heads - num_inter_heads1

    A = torch.zeros(H, N, N, device=device)

    ################################
    # 1) Domain1 heads => [0..H1)
    ################################
    if subblock in ('all','domain1'):
        for h in range(model.H1):
            # local h = h
            Q_sel = model.Q1[h]  # shape (d, L1)
            K_sel = model.K1[h]  # shape (d, L1)
            e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)  # (L1, L1)
            sf = F.softmax(e_sel, dim=1)
            A[h, :L1, :L1] = sf

        # Zero diagonal for domain1
        for h in range(model.H1):
            A[h, :L1, :L1].fill_diagonal_(0)

    ################################
    # 2) Domain2 heads => [H1..H2)
    ################################
    if subblock in ('all','domain2'):
        for h in range(model.H1, model.H2):
            local_idx = h - model.H1
            Q_sel = model.Q2[local_idx]  # shape (d, L2)
            K_sel = model.K2[local_idx]  # shape (d, L2)
            e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)  # (L2, L2)
            sf = F.softmax(e_sel, dim=1)
            A[h, dstart:, dstart:] = sf

        # Zero diagonal for domain2
        for h in range(model.H1, model.H2):
            A[h, dstart:, dstart:].fill_diagonal_(0)

    ######################################
    # 3) Inter heads => [H2..H) total
    #    a) domain1->domain2 => [H2..(H2+num_inter_heads1))
    #    b) domain2->domain1 => [H2+num_inter_heads1..H)
    ######################################
    if subblock in ('all','inter'):
        # (a) domain1->domain2 heads
        for h in range(model.H2, model.H2 + num_inter_heads1):
            local_idx = h - model.H2
            Q_sel = model.Qint1[local_idx]  # shape (d, L1)
            K_sel = model.Kint1[local_idx]  # shape (d, L2)
            e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)  # (L1, L2)
            sf = F.softmax(e_sel, dim=1)
            A[h, :L1, dstart:] = sf

            if make_inter_sym:
                # Mirror domain1->domain2 into domain2->domain1
                A[h, dstart:, :L1] = sf.transpose(0,1)

        # (b) domain2->domain1 heads
        for h in range(model.H2 + num_inter_heads1, model.H2 + num_inter_heads1 + num_inter_heads2):
            local_idx = h - (model.H2 + num_inter_heads1)
            Q_sel = model.Qint2[local_idx]  # shape (d, L2)
            K_sel = model.Kint2[local_idx]  # shape (d, L1)
            e_sel = torch.einsum('di,dj->ij', Q_sel, K_sel)  # (L2, L1)
            sf = F.softmax(e_sel, dim=1)

            # Place domain2->domain1 in bottom-left:
            #   row dimension => domain2
            #   col dimension => domain1
            # so shape is (L2, L1).
            A[h, dstart:, :L1] = sf

            if make_inter_sym:
                # If you want to mirror domain2->domain1 into domain1->domain2, 
                # i.e. top-right block:
                # Typically you'd do:
                A[h, :L1, dstart:] = sf.transpose(0,1)

    return A


###############################################################################
#  2) Reuse your existing "k_matrix_precomputed" logic
###############################################################################
def k_matrix_precomputed(A, k, version='mean', sym=True, APC=False, sqr=False):
    """
    Same logic as k_matrix, but we assume `A` is already shape (H, N, N)
    i.e. the full attention map across all heads.
    """
    device = A.device
    H, N, _ = A.shape
    if sqr:
        A = A * A  # Element-wise square
    
    if k >= N * (N - 1) // 2:
        if version == 'mean':
            M = torch.mean(A, dim=0)  # Shape: (N, N)
        elif version == 'maximum':
            M, _ = torch.max(A, dim=0)  # Shape: (N, N)
        
        if sym:
            M = 0.5 * (M + M.transpose(0,1))
        
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
        mask = (count_A > 0)
        M[mask] = sum_A[mask] / count_A[mask]
        
    if sym:
        M = 0.5 * (M + M.transpose(0,1))
    
    if APC:
        M = correct_APC(M)
    
    return M, _A

###############################################################################
#  3) A helper for reading the true structure
###############################################################################
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
    dist_dict = compute_residue_pair_dist(structfile)  # dict with keys=(i,j), values=distance
    
    filtered_dist = {
        k: v for k, v in dist_dict.items()
        if (k[1] - k[0] > min_separation) and (v <= cutoff) and (v != 0)
    }
    
    if not filtered_dist:
        return np.array([[], []])
    
    pair_indices = list(filtered_dist.keys())
    coords = np.array(pair_indices).T  # shape: (2, M)
    return coords

###############################################################################
#  4) Final plotting function that displays the aggregated matrix M
#     plus optionally iterates over A's heads if "all=True"
###############################################################################
def graphAtt_precomputed(
    M, A,
    filestruct,
    PFname,
    ticks,
    k=None,
    version='mean',
    sqr=False,
    APC=True,
    all=False
):
    """
    Visualizes the aggregated contact map M (shape N x N),
    along with optionally each head in A (H x N x N).

    Parameters:
      - M: (N, N) aggregated contact map
      - A: (H, N, N) full stack of heads
      - filestruct: PDB structure file or some distance data for plotting true structure
      - PFname: Title string for the plot
      - ticks: Tick positions (for labeling the axes)
      - k, version, sqr, APC: carried from the context
      - all: if True, also loop over each head in A and plot it individually
    """
    N = A.shape[1]
    ms = 100 / N

    # 1) Optionally loop over each head
    if all:
        for i in range(A.shape[0]):
            A_i = A[i].clone()  # (N, N)
            mean_val = torch.mean(A_i)
            # fill diagonal with mean
            for j in range(A_i.size(0)):
                A_i[j, j] = mean_val
            A_i_np = A_i.cpu().detach().numpy()
    
            plt.figure()
            plt.imshow(A_i_np, cmap='gray_r', aspect='auto')
            plt.colorbar(shrink=0.7, aspect=10)

            dist = true_structure(filestruct, min_separation=0, cutoff=8.0)
            if dist.size != 0:
                plt.scatter((dist[0] - 1), (dist[1] - 1),
                            c='r', marker='o', alpha=0.1, s=ms, label='True Structure')
                plt.scatter((dist[1] - 1), (dist[0] - 1),
                            c='r', marker='o', alpha=0.1, s=ms)
            plt.xticks(ticks - 1, ticks, fontsize=5)
            plt.yticks(ticks - 1, ticks, fontsize=5)
            plt.title(f"{PFname}, Head {i+1}")
            plt.legend()
            plt.show()

    # 2) Plot the aggregated matrix M
    # replace diagonal with its mean
    mean_M = torch.mean(M)
    M_plot = M.clone()
    for i in range(M_plot.size(0)):
        M_plot[i, i] = mean_M

    M_np = M_plot.cpu().detach().numpy()
    plt.figure()
    plt.imshow(M_np, cmap='gray_r')
    plt.colorbar(shrink=0.7, aspect=10)

    dist = true_structure(filestruct, min_separation=0, cutoff=8.0)
    if dist.size != 0:
        plt.scatter((dist[0] - 1), (dist[1] - 1),
                    c='r', marker='o', alpha=0.1, s=ms, label='True Structure')
        plt.scatter((dist[1] - 1), (dist[0] - 1),
                    c='r', marker='o', alpha=0.1, s=ms)
    plt.xticks(ticks - 1, ticks, fontsize=5)
    plt.yticks(ticks - 1, ticks, fontsize=5)
    plt.title(f"{PFname}, {version} Aggregated", fontsize=12)
    plt.legend()
    plt.show()