import numpy as np
import torch




def compute_fn(J):
    """
    Computes the Frobenius norm for each residue pair.

    Parameters:
    - J: Tensor of shape (q, q, L, L)

    Returns:
    - fn: Tensor of shape (L, L)
    """
    q, _, L, _ = J.shape
    # Select slices from 0 to q-2 (since Python is 0-indexed)
    J_reduced = J[:q - 1, :q - 1, :, :]  # Shape: (q-1, q-1, L, L)
    # Compute sum over a and b (first two dimensions)
    fn = torch.sum(J_reduced ** 2, dim=(0, 1))  # Shape: (L, L)
    # Symmetrize fn
    fn = (fn + fn.t()) * 0.5
    return fn

def correct_APC(S):
    """
    Corrects the S matrix using Average Product Correction (APC).

    Parameters:
    - S: Tensor of shape (L, L)

    Returns:
    - S_corrected: APC-corrected tensor
    """
    N = S.shape[0]
    Si = torch.sum(S, dim=0, keepdim=True)  # Shape: (1, L)
    Sj = torch.sum(S, dim=1, keepdim=True)  # Shape: (L, 1)
    Sa = torch.sum(S) * (1 - 1 / N)
    S_corrected = S - (Sj @ Si) / Sa  # Outer product
    return S_corrected

def compute_ranking(S, min_separation=6):
    """
    Computes the ranking of residue pairs based on the S matrix.

    Parameters:
    - S: Tensor of shape (L, L)
    - min_separation: Minimum separation between residues

    Returns:
    - R: List of tuples (i, j, S[j, i]) sorted by S[j, i] in descending order
    """
    N = S.shape[0]
    # Get indices where j - i >= min_separation
    i_indices, j_indices = torch.triu_indices(N, N, offset=min_separation)
    # Adjust to 1-based indexing to match Julia's indexing
    i_indices_1b = i_indices + 1
    j_indices_1b = j_indices + 1
    # Extract scores
    scores = S[j_indices, i_indices]
    # Combine into list of tuples
    R = list(zip(i_indices_1b.tolist(), j_indices_1b.tolist(), scores.tolist()))
    # Sort by score descending
    R.sort(key=lambda x: x[2], reverse=True)
    return R

class Gauge:
    pass

class ZeroSumGauge(Gauge):
    pass

def shift(hT, gauge):
    """
    Compute the shift vector for ZeroSumGauge.

    Parameters:
    - hT: numpy array of shape (q, N)
    - gauge: an instance of ZeroSumGauge

    Returns:
    - vecshift: numpy array of length N
    """
        
    if isinstance(gauge, ZeroSumGauge):
        # For ZeroSumGauge, shift is the mean over 'a' for each 'i'
        vecshift = np.mean(hT, axis=0)  # Shape: (N,)
        return vecshift

    else:
        raise NotImplementedError("Gauge type not implemented")

def UV(J):
    """
    Compute U and V for the ZeroSumGauge.

    Parameters:
    - J: numpy array of shape (q, q, N, N)

    Returns:
    - U: numpy array of shape (q, N, N)
    - V: numpy array of shape (q, N, N)
    """
    q, _, N, _ = J.shape
    U = np.zeros((q, N, N))
    V = np.zeros((q, N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                J_ij = J[:, :, i, j]
                Jss = np.mean(J_ij)

                # Mean over columns (axis=1)
                mean_over_cols = np.mean(J_ij, axis=1)
                V[:, i, j] = -mean_over_cols + 0.5 * Jss

                # Mean over rows (axis=0)
                mean_over_rows = np.mean(J_ij, axis=0)
                U[:, i, j] = -mean_over_rows + 0.5 * Jss
    return U, V

def gauge(J, h, gauge, return_h=False):
    """
    Apply the gauge transformation to J and h according to the specified gauge.

    Parameters:
    - J: numpy array of shape (q, q, N, N)
    - h: numpy array of shape (q, N)
    - gauge: an instance of a Gauge subclass

    Returns:
    - JT: transformed J
    - hT: transformed h
    """
    # Check that J is symmetric: J[a, b, i, j] == J[b, a, j, i]
    if not np.allclose(J, np.transpose(J, (1, 0, 3, 2)), atol=1e-10):
        raise ValueError("J should be symmetric: J != J.transpose(1, 0, 3, 2)")

    q, _, N, _ = J.shape

    # Compute U and V according to the gauge
    if isinstance(gauge, ZeroSumGauge):
        U, V = UV(J)
    else:
        raise NotImplementedError("Gauge type not implemented")

    # # Compute JT
    # JT = np.zeros_like(J)
    # for i in range(N):
    #     for j in range(N):
    #         if i != j:
    #             for a in range(q):
    #                 for b in range(q):
    #                     JT[b, a, j, i] = J[b, a, j, i] + V[a, i, j] + U[b, i, j]
    #         else:
    #             # Diagonal elements remain the same
    #             JT[:, :, i, i] = J[:, :, i, i]

    # Expand U and V to enable broadcasting
    U_broadcasted = U[:, np.newaxis, :, :]  # Shape: (q, 1, N, N)
    V_broadcasted = V[np.newaxis, :, :, :]  # Shape: (1, q, N, N)

    # Compute JT
    JT = J + U_broadcasted + V_broadcasted  # Shape: (q, q, N, N)

    # Set diagonal elements back to original J
    idx = np.arange(N)
    JT[:, :, idx, idx] = J[:, :, idx, idx]

    if return_h:
        # Compute hT
        hT = np.copy(h)
        for i in range(N):
            for a in range(q):
                for j in range(i + 1, N):
                    hT[a, i] -= V[a, i, j]
                for j in range(i):
                    hT[a, i] -= U[a, j, i]

        # Apply shift to hT
        vecshift = shift(hT, gauge)
        for i in range(N):
            hT[:, i] -= vecshift[i]
    else:
        hT = 0
    return JT, hT

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

def score(
    model,
    Q, K, V,
    min_separation=6,
    index_last_domain1=0,
    H1=0,
    H2=0,
    A=None,
    nb_pred=None
):
    """
    Computes a ranking/score for the coupling J in the model, automatically 
    using model.compute_attention_heads(...) for domain-split or no-split attention.

    Parameters
    ----------
    model : nn.Module
        An instance of your AttentionModel (or any class with compute_attention_heads).
    Q : torch.Tensor
        Shape (H, d, L). The Q parameter.
    K : torch.Tensor
        Shape (H, d, L). The K parameter.
    V : torch.Tensor
        Shape (H, q, q). The V parameter.
    min_separation : int
        Minimum separation for the final ranking.
    index_last_domain1 : int
        0 => no domain-split; > 0 => domain-split.
    H1 : int
        Number of heads for domain 1.
    H2 : int
        Number of heads for domain 2.
    A : torch.Tensor or None
        If not None, must be shape (H, L, L). The user’s precomputed attention heads.

    Returns
    -------
    ranking : np.ndarray
        1D array of contact pairs in sorted order (best to worst).
    JT_np : np.ndarray
        The final J coupling tensor in zero-sum gauge, shape (q, q, L, L).

    Notes
    -----
    - If A is provided, we skip computing the heads and use A as the attention matrix.
    - Otherwise, attention is always computed from model.compute_attention_heads(...),
      which handles domain-split logic automatically (via index_last_domain1, H1, H2).
    """
    device = Q.device
    L = Q.shape[-1]  # e.g. number of positions/residues

    ###########################################################################
    # 1) Get attention matrix W of shape (H, L, L)
    ###########################################################################
    if A is not None:
        # Use the provided attention matrix
        W = A  
        # Make sure it's on the same device as Q (if needed)
        if W.device != device:
            W = W.to(device)
    else:
        # Let the model compute the heads for us
        W = attention_heads_from_model(
            model, Q, K, V,
            index_last_domain1=index_last_domain1,
            H1=H1,
            H2=H2,
            sym=False
        )

    # Mask out the diagonal (i != j)
    i_indices = torch.arange(L, device=device).unsqueeze(1)
    j_indices = torch.arange(L, device=device).unsqueeze(0)
    mask = (i_indices != j_indices).float().unsqueeze(0)  # shape (1, L, L)
    W = W * mask
        
    # Compute Jtens
    Jtens = torch.einsum('hri,hab->abri', W, V)  # Shape: (q, q, L, L)
    q = Jtens.shape[0]
    N = Jtens.shape[2]

    # Compute Jtens
    Jtens = torch.einsum('hri,hab->abri', W, V)  # Shape: (q, q, L, L)

    # Ensure Jtens is on CPU and detached from any computation graph
    Jtens_cpu = Jtens.cpu().detach()

    # Convert to NumPy array with dtype float64
    Jtens_np = Jtens_cpu.numpy().astype(np.float64)

    J = 0.5 * (Jtens_np + np.transpose(Jtens_np, (1, 0, 3, 2)))  # Make symmetric
    
    del Jtens, Jtens_cpu, Jtens_np

    #h = np.random.randn(q, N)
    h = np.zeros((q, N))

    # Instantiate the ZeroSumGauge
    gauge_type = ZeroSumGauge()

    JT, _ = gauge(J, h, gauge_type)

    JT_np = np.array(JT)

    # Convert to PyTorch tensor
    Jzsg_tensor = torch.from_numpy(JT_np).to(device)

    # Compute Frobenius norms
    FN = compute_fn(Jzsg_tensor)

    del Jzsg_tensor

    # Correct for Average Product Correction
    FNapc = correct_APC(FN)

    del FN

    if nb_pred is None:
        nb_pred = L * (L - 1) // 2 #TODO: check that it is correct

    # Compute ranking
    ranking = compute_ranking(FNapc, min_separation)[:nb_pred]
    return ranking, JT_np

def score_multiple_realizations(Qs, Ks, Vs, min_separation=6, nb_pred=None):

    H, d, L = Qs[0].shape
    _, q, _ = Vs[0].shape
    device = Qs.device

    N = L

    nb_reps=Qs.shape[0]

    Fs=np.zeros((nb_reps,L,L))
    for rep in range(nb_reps):
        Q=Qs[rep]
        K=Ks[rep]
        V=Vs[rep]
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

        # Compute Jtens
        Jtens = torch.einsum('hri,hab->abri', W, V)  # Shape: (q, q, L, L)

        # Ensure Jtens is on CPU and detached from any computation graph
        Jtens_cpu = Jtens.cpu().detach()

        # Convert to NumPy array with dtype float64
        Jtens_np = Jtens_cpu.numpy().astype(np.float64)

        J = 0.5 * (Jtens_np + np.transpose(Jtens_np, (1, 0, 3, 2)))  # Make symmetric

        del Jtens, Jtens_cpu, Jtens_np

        # Random h vector
        #h = np.random.randn(q, N)
        h = np.zeros((q, N))

        # Instantiate the ZeroSumGauge
        gauge_type = ZeroSumGauge()

        JT, _ = gauge(J, h, gauge_type)

        JT_np = np.array(JT)

        # Convert to PyTorch tensor
        Jzsg_tensor = torch.from_numpy(JT_np).to(device)

        # Compute Frobenius norms
        FN = compute_fn(Jzsg_tensor)

        del Jzsg_tensor

        # Correct for Average Product Correction
        Fs[rep] = correct_APC(FN)

    # Take the maximum value for each i, j across all realizations
    FNapc = np.max(Fs, axis=0)

    del FN, Fs
    
    if nb_pred is None:
        nb_pred = L * (L - 1) // 2 #TODO: check that it is correct

    # Compute ranking
    ranking = compute_ranking(FNapc, min_separation)[:nb_pred]
    return ranking, JT_np

def compute_residue_pair_dist(filedist):
    """
    Reads a structure file and computes residue pair distances.

    Parameters:
    - filedist: Path to the structure file

    Returns:
    - A dictionary with keys (sitei, sitej) and values as distances
    """
    import numpy as np
    d = np.loadtxt(filedist)
    if d.shape[1] == 4:
        return {
            (int(round(row[0])), int(round(row[1]))): row[3]
            for row in d if row[3] != 0
        }
    elif d.shape[1] == 3:
        return {
            (int(round(row[0])), int(round(row[1]))): row[2]
            for row in d if row[2] != 0
        }
    else:
        raise ValueError("Unexpected number of columns in the structure file.")
    
def compute_referencescore(score, dist, mindist=6, cutoff=8.0):
    """
    Computes reference scores and PPV values.

    Parameters:
    - score: List of tuples (sitei, sitej, plmscore)
    - dist: Dictionary of distances between residue pairs
    - mindist: Minimum separation between residues
    - cutoff: Distance cutoff to consider a contact

    Returns:
    - List of tuples (sitei, sitej, plmscore, PPV)
    """
    nc2 = len(score)
    out = []
    ctrtot = 0
    ctr = 0
    for i in range(nc2):
        sitei, sitej, plmscore = score[i]
        if (sitei, sitej) in dist:
            dij = dist[(sitei, sitej)]
        elif (sitej, sitei) in dist:
            dij = dist[(sitej, sitei)]
        else:
            continue
        if abs(sitej - sitei) >= mindist:
            ctrtot += 1
            if dij < cutoff:
                ctr += 1
            PPV = ctr / ctrtot
            out.append((sitei, sitej, plmscore, PPV))
    return out

def compute_referencescore_from_map(score, distance_map, min_separation=6, cutoff=8.0):
    """
    Computes reference scores and PPV values.

    Parameters:
    - score: List of tuples (sitei, sitej, plmscore) (1-indexed) #TODO: change to 0-indexed
    - distance_map: LxL matrix with distances between residue pairs
    - min_separation: Minimum separation between residues
    - cutoff: Distance cutoff to consider a contact

    Returns:
    - List of tuples (sitei, sitej, plmscore, PPV)
    """
    nc2 = len(score)
    out = []
    ctrtot = 0
    ctr = 0

    for i in range(nc2):
        sitei, sitej, plmscore = score[i]
        sitei=int(sitei)
        sitej=int(sitej)
        dij = distance_map[sitei-1, sitej-1] # -1 to match 0-indexing TODO: change when changing score

        if abs(sitej - sitei) >= min_separation:
            ctrtot += 1
            if dij < cutoff:
                ctr += 1
            PPV = ctr / ctrtot

            out.append((sitei, sitej, plmscore, PPV))

    return out


def compute_PPV(score, filestruct, min_separation=6, cutoff=8.0):
    """
    Computes the Positive Predictive Value (PPV).

    Parameters:
    - score: List of tuples (sitei, sitej, plmscore)
    - filestruct: Path to the structure file
    - min_separation: Minimum separation between residues

    Returns:
    - List of PPV values
    """
    dist = compute_residue_pair_dist(filestruct)
    ref_scores = compute_referencescore(score, dist, mindist=min_separation, cutoff=cutoff, nb_pred=nb_pred)
    return [x[3] for x in ref_scores]


def compute_PPV_from_map(score, distancemap_file, min_separation=6, cutoff=8.0, nb_pred=100): #TODO: rewrite
    """
    Computes the Positive Predictive Value (PPV).

    Parameters:
    - score: List of tuples (sitei, sitej, plmscore)
    - distancemap_file: path to distance map file (LxL matrix with distances between each pair of residues)
    - min_separation: Minimum separation between residues

    Returns:
    - List of PPV values
    """
    if nb_pred>len(score):
        nb_pred=len(score)
    distance_map = np.loadtxt(distancemap_file)
    ref_scores = compute_referencescore_from_map(score[:nb_pred], distance_map, min_separation=min_separation, cutoff=cutoff)
    return [x[3] for x in ref_scores]




