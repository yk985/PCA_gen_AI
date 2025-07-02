import numpy as np
import torch
import matplotlib.pyplot as plt

from CODE.AttentionDCA_python.src.dcascore import  correct_APC, compute_residue_pair_dist
# Import your model so we can call create_attention_masks if truly needed
from CODE.AttentionDCA_python.src.model import AttentionModel  # or however your relative import is
####################################################################

def attention_heads_from_model(model, Q, K, V, index_last_domain1=0, H1=0, H2=0, sym=False):
    """
    Use the model's compute_attention_heads(...) to get the final attention heads.
    This removes duplication of the domain mask logic.
    
    Returns:
      A -- shape: (H, N, N)
    """
    # sf will have shape (N, N, H)
    sf = model.compute_attention_heads(
        Q, K, V,
        index_last_domain1=index_last_domain1, 
        H1=H1, 
        H2=H2
    )
    # Permute to get (H, N, N)
    A = sf.permute(2, 0, 1)
    if sym:
        # Symmetrize
        A = 0.5 * (A + A.transpose(1, 2))
    return A

def true_structure(structfile, min_separation=0, cutoff=8.0):
    """
    Compute the true structure coordinates based on residue pair distances.
    """
    dist_dict = compute_residue_pair_dist(structfile)  
    filtered_dist = {
        k: v for k, v in dist_dict.items()
        if (k[1] - k[0] > min_separation) and (v <= cutoff) and (v != 0)
    }
    if not filtered_dist:
        return np.array([[], []])
    
    pair_indices = list(filtered_dist.keys())
    coords = np.array(pair_indices).T  # shape: (2, M)
    return coords

def k_matrix(
    model,
    Q, K, V,
    k, version='mean', 
    sym=True, APC=False, sqr=False,
    index_last_domain1=0, H1=0, H2=0
):
    """
    Produces a 2D contact map from the heads.
    Uses attention_heads_from_model(...) under the hood.
    """
    if version not in ['mean', 'maximum']:
        raise ValueError("Only 'mean' or 'maximum' versions are supported.")
    
    # Get raw attention heads (H, N, N) via the model
    A = attention_heads_from_model(
        model, Q, K, V,
        index_last_domain1=index_last_domain1,
        H1=H1, 
        H2=H2,
        sym=sym
    )
    H, N, _ = A.shape

    # Optionally square
    if sqr:
        A = A * A

    # If k is large (>= total number of pairs), just do the global mean or max
    if k >= (N*(N-1)//2):
        if version == 'mean':
            M = torch.mean(A, dim=0)  # (N, N)
        else:  # 'maximum'
            M, _ = torch.max(A, dim=0)  # (N, N)
        M = 0.5*(M + M.T)  # symmetrize

        if APC:
            M = correct_APC(M)
        return M, A

    # Else, do the top-k approach per head
    _A = torch.zeros_like(A)
    for h in range(H):
        A_h = A[h].flatten()  # shape: (N*N,)
        top_vals, idxs = torch.topk(A_h, k, largest=True, sorted=False)
        i_indices = idxs // N
        j_indices = idxs % N
        _A[h, i_indices, j_indices] = top_vals

    if version == 'maximum':
        M, _ = torch.max(_A, dim=0)
    else:  # 'mean'
        sum_A = _A.sum(dim=0)
        count_A = (_A != 0).sum(dim=0)
        M = torch.zeros_like(sum_A)
        mask = (count_A > 0)
        M[mask] = sum_A[mask] / count_A[mask]

    M = 0.5*(M + M.T)

    if APC:
        M = correct_APC(M)
    return M, _A

def graphAtt(
    model,
    Q, K, V,
    filestruct,
    PFname,
    ticks,
    k=None,
    version='mean',
    sqr=False,
    APC=True,
    show_all_heads=False,
    index_last_domain1=0,
    H1=0,
    H2=0
):
    """
    Visualizes the Attention Map based on Q, K, V matrices using the 
    model's compute_attention_heads to avoid duplication.
    """
    # Calculate marker size
    ms = 100 / Q.shape[2]

    # If k is None, pick some default
    if k is None:
        k_default = Q.shape[0]  # e.g., # heads
        M, A = k_matrix(
            model, Q, K, V,
            k=k_default, 
            version=version, 
            sym=True, 
            APC=APC, 
            sqr=sqr,
            index_last_domain1=index_last_domain1,
            H1=H1,
            H2=H2
        )
    else:
        M, A = k_matrix(
            model, Q, K, V,
            k=k, 
            version=version, 
            sym=True, 
            APC=APC, 
            sqr=sqr,
            index_last_domain1=index_last_domain1,
            H1=H1,
            H2=H2
        )

    # Optionally show all heads individually
    if show_all_heads:
        # A has shape (H, N, N)
        for i in range(A.shape[0]):
            plt.figure()
            # Replace diagonal with mean
            A_i = A[i].clone()
            mean_val = torch.mean(A_i)
            for j in range(A_i.size(0)):
                A_i[j, j] = mean_val
            Am_i = A_i.cpu().detach().numpy()
            plt.imshow(Am_i, cmap='gray_r')
            plt.title(f"{PFname}, Head {i+1}")
            plt.colorbar()
            plt.xticks(ticks - 1, ticks, fontsize=5)
            plt.yticks(ticks - 1, ticks, fontsize=5)

            if filestruct is not None:
                dist = true_structure(filestruct, min_separation=0, cutoff=8.0)
                if dist.size != 0:
                    plt.scatter((dist[0] - 1), (dist[1] - 1), 
                                c='r', marker='o', alpha=0.1, s=ms, label='True Structure')
                    plt.scatter((dist[1] - 1), (dist[0] - 1), 
                                c='r', marker='o', alpha=0.1, s=ms)
            plt.legend()
            plt.show()

    # Now plot the aggregated matrix M
    # Replace diagonal elements with the mean
    mean_M = torch.mean(M)
    for i in range(M.size(0)):
        M[i, i] = mean_M

    Am = M.cpu().detach().numpy()
    plt.figure()
    plt.imshow(Am, cmap='gray_r')
    plt.colorbar()
    plt.title(f"{PFname}, {version} Aggregated", fontsize=12)
    plt.xticks(ticks - 1, ticks, fontsize=5)
    plt.yticks(ticks - 1, ticks, fontsize=5)

    if filestruct is not None:
        dist = true_structure(filestruct, min_separation=0, cutoff=8.0)
        if dist.size != 0:
            plt.scatter((dist[0] - 1), (dist[1] - 1), 
                        c='r', marker='o', alpha=0.1, s=ms, label='True Structure')
            plt.scatter((dist[1] - 1), (dist[0] - 1), 
                        c='r', marker='o', alpha=0.1, s=ms)
    plt.legend()
    plt.show()