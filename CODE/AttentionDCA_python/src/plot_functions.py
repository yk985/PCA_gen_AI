import numpy as np
import torch
import matplotlib.pyplot as plt
from CODE.AttentionDCA_python.src.dcascore import compute_referencescore, correct_APC, compute_residue_pair_dist, compute_PPV_from_map
from CODE.AttentionDCA_python.src.model import AttentionModel
plt.rc('font',size=12)

def attention_heads(Q, K, V, sym=False, head_mask=0, H1 = 0, H2 = 0):
    double_domain_masks = head_mask
    H, d, N = Q.shape
    device = Q.device  # Ensure consistency with device
    mask_value = -1e9  # A large negative value to zero out after softmax

    if head_mask is not None:
        e = torch.einsum('hdi,hdj->ijh', Q, K)  # Shape: (N, N, H)

        # Exclude self-interactions by setting scores to -inf on the diagonal
        i_indices = torch.arange(N, device=device).unsqueeze(1)
        j_indices = torch.arange(N, device=device).unsqueeze(0)
        self_mask = (i_indices != j_indices).float()
        mask_value = -1e9  # A large negative value to zero out after softmax
        e = e * self_mask.unsqueeze(-1) + (1 - self_mask.unsqueeze(-1)) * mask_value

        # Apply domain masks if specified
        if double_domain_masks > 0 and double_domain_masks < N:
            model = AttentionModel(H,d,N,q)
            domain_masks= model.create_attention_masks(H, N, double_domain_masks, double_domain_masks + 1,H1=H1, H2=H2, device= device)
            # Invert the domain masks to identify positions to mask
            inverted_domain_masks = (1 - domain_masks).bool()  # Positions to mask are True
            # Permute e to match the shape of domain_masks
            e = e.permute(2, 0, 1)  # Shape: (H, N, N)
            # Apply the masks
            e = e.masked_fill(inverted_domain_masks, mask_value)
            # Permute e back to original shape
            e = e.permute(1, 2, 0)  # Shape: (N, N, H)
        H, q, _ = V.shape
        N_e1, N_e2, H_e = e.shape
        device = e.device
        dtype = e.dtype
        domain1_end = double_domain_masks
        domain2_start = double_domain_masks + 1    

        sf = torch.zeros(N, N, H, device=device, dtype=dtype)
        for h in range(H):
            if domain1_end != 0:
                if h < H1:
                    # Heads for Domain 1
                    softmax_vals = torch.softmax(e[0:domain1_end+1, 0:domain1_end+1, h], dim=1)
                    top = torch.cat([
                        softmax_vals,
                        torch.zeros(domain1_end+1, N - domain1_end-1, device=device)
                    ], dim=1)
                    sf_domain = torch.cat([
                        top,
                        torch.zeros(N - domain1_end-1, N, device=device)
                    ], dim=0)
                    sf = sf.clone()
                    sf[:, :, h] = sf_domain
                elif h < H2:
                    # Heads for Domain 2
                    softmax_vals = torch.softmax(e[domain2_start:, domain2_start:, h], dim=1)
                    # Create the top-left zero block

                    bottom_left = torch.zeros(N - domain2_start, domain2_start, device=device)  # 111x65


                    # Concatenate top and bottom parts
                    top = torch.zeros(domain2_start, N, device=device)     # 65x176
                    bottom = torch.cat([bottom_left, softmax_vals], dim=1)  # 111x176

                    # Concatenate top and bottom to form the final sf_domain
                    sf_domain = torch.cat([top, bottom], dim=0)        # 176x176

                    # Assign sf_domain to the corresponding head in sf
                    sf = sf.clone()
                    sf[:, :, h] = sf_domain

                else:
                    # Heads for inter-domain interactions
                    sf_domain = torch.softmax(e[:, :, h], dim=1)
                    sf = sf.clone()
                    sf[:, :, h] = sf_domain
            else:
                # No domain masks applied
                sf_domain = torch.softmax(e[:, :, h], dim=1)
                sf = sf.clone()
                sf[:, :, h] = sf_domain

        A = sf.clone()
        A = A.permute(2, 0, 1)
        # Apply softmax over the last dimension (dim=2)
        #A = torch.softmax(e, dim=2)  # Shape: (H, N, N)
    else:
        # Original computation
        W = torch.einsum('hdi, hdj -> hij', Q, K)  # Shape: (H, N, N)
        A = torch.softmax(W, dim=2)  # Shape: (H, N, N)
        # Apply mask to exclude self-interactions
        mask = ~torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0)  # Shape: (1, N, N)
        A = A * mask  # Mask diagonal

    if sym:
        A = (A + A.transpose(1, 2)) / 2

    return A  # Shape: (H, N, N)

def attention_heads_interaction(Q, K, V, sym=False):

    W = torch.einsum('hdi, hdj -> hij', Q, K)  # Shape: (H, L_A, L_B)
    A = torch.softmax(W, dim=2)  # Softmax over j

    return A  

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


def k_matrix(Q, K, V, k, version, sym=True, APC=False, sqr=False):
    if version not in ['mean', 'maximum']:
        raise ValueError("Only 'mean' and 'maximum' versions are supported.")
    
    A = attention_heads(Q, K, V, sym=sym)  # Shape: (H, N, N)
    H, N, _ = A.shape
    M = torch.zeros(N, N, device=A.device, dtype=A.dtype)
    
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


def k_matrix_interaction(Q, K, V, k, version, sym=True, APC=False, sqr=False):
    if version not in ['mean', 'maximum']:
        raise ValueError("Only 'mean' and 'maximum' versions are supported.")
    
    A = attention_heads_interaction(Q, K, V, sym=sym)  # Shape: (H, N, N)
    H, L_A, L_B = A.shape
    M = torch.zeros(L_A, L_B, device=A.device, dtype=A.dtype)
    
    if sqr:
        A = A * A  # Element-wise square
    
    if k >= L_A * (L_A - 1) / 2: 
        print("caution needed")
        if version == 'mean':
            M = torch.mean(A, dim=0)  # Shape: (N, N)
        elif version == 'maximum':
            M, _ = torch.max(A, dim=0)  # Shape: (N, N)
        
        M = (M + M.transpose(0, 1)) / 2  # Symmetrize
        
        if APC:
            M = correct_APC(M)
        
        return M, A
    
    _A = torch.zeros(H, L_A, L_B, device=A.device, dtype=A.dtype)
    N=L_A+L_B
    for h in range(H):
        # Flatten the h-th attention matrix
        A_h = A[h].flatten()  # Shape: (L_A*L_B,)
        
        # Get top-k values and their indices
        vmins, idxs = torch.topk(A_h, k, largest=True, sorted=False)  # Shape: (k,)
        print("idxs:")
        print(idxs)

        # Convert flat indices to 2D indices
        i_indices = idxs // L_B
        j_indices = idxs % L_B
        
        # Assign the top-k values to _A
        _A[h, i_indices, j_indices] = vmins
    
    if version == 'maximum':
        M, _ = torch.max(_A, dim=0)  # Shape: (N, N)
    elif version == 'mean':
        sum_A = _A.sum(dim=0)  # Shape: (N, N)
        count_A = (_A != 0).sum(dim=0)  # Shape: (N, N)
        M = torch.zeros(L_A, L_B, device=A.device, dtype=A.dtype)
        mask = count_A > 0
        M[mask] = sum_A[mask] / count_A[mask]
        
    
    #M = (M + M.transpose(0, 1)) / 2  # Symmetrize
    
    if APC:
        M = correct_APC(M)
    
    return M, _A

def graphAtt(Q, K, V, filestruct, PFname, ticks, k=None, version='mean', sqr=False, APC=True, all = False, sym=True):
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
    # Calculate marker size
    ms = 100 / Q.shape[2]
    
    # Compute Am using k_matrix function
    if k is None:
        # If k is not provided, set it to the maximum possible
        k_default = Q.shape[0]  # Example default value, adjust as needed
        M, A = k_matrix(Q, K, V, k=k_default, version=version, sym=sym, APC=APC, sqr=sqr)
    else:
        M, A = k_matrix(Q, K, V, k=k, version=version, sym=sym, APC=APC, sqr=sqr)
    
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
    # Add colorbar with specified shrink and aspect
    plt.savefig('./figs/attmap_{PFname}.pdf'.format(PFname=PFname))
    # Add legend
    plt.legend()


def graphAtt_interaction(Q, K, V, filestruct, PFname, ticks, k=None, version='mean', sqr=False, APC=True, all = False, sym=True):
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
    # Calculate marker size
    ms = 100 / Q.shape[2]
    
    # Compute Am using k_matrix function
    if k is None:
        # If k is not provided, set it to the maximum possible
        k_default = Q.shape[0]  # Example default value, adjust as needed
        M, A = k_matrix_interaction(Q, K, V, k=k_default, version=version, sym=sym, APC=APC, sqr=sqr)
    else:
        M, A = k_matrix_interaction(Q, K, V, k=k, version=version, sym=sym, APC=APC, sqr=sqr)
    
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
    # Add colorbar with specified shrink and aspect
    
    # Add legend
    plt.legend()

def graphPPV(PPVs, labels, figtitle,
            fig_size=(8, 6),
            colors=["r", "b", "g", "y"],
            fs=15):
    """
    Plot PPV sequences on a semi-logarithmic plot.

    Parameters:
    - PPVs (list of lists or numpy arrays): List containing PPV sequences.
    - labels (list of str): List of labels for the PPV sequences (excluding the first "Structure" label).
    - figtitle (str): Title of the plot.
    - fig_size (tuple, optional): Size of the figure in inches. Default is (8, 6).
    - colors (list of str, optional): List of colors for additional PPV sequences. Default is ["r", "b", "g", "y"].
    - fs (int, optional): Font size for labels and ticks. Default is 15.
    """
    # Validate inputs
    l = len(PPVs)
    if l > 1 and len(labels) != l - 1:
        raise ValueError("Number of labels must be equal to number of PPV sequences minus one.")
    
    # Close all existing figures
    #plt.close('all')
    # Create a new figure with specified size
    plt.figure(figsize=fig_size)
    
    if l > 1:
        # The first PPV is the structure
        PPV_structure = PPVs[0]
        x0 = np.arange(1, len(PPV_structure) + 1)
        plt.semilogx(x0, PPV_structure, ".-", markersize=1, label="Structure", color="gray")
        
        # The second PPV is from PlmDCA
        PPV_plmDCA = PPVs[1]
        x1 = np.arange(1, len(PPV_plmDCA) + 1)
        plt.semilogx(x1, PPV_plmDCA, ".-", label=labels[0], color="k")
        
        # Plot additional PPVs
        for i in range(l - 2):
            PPV_additional = PPVs[i + 2]
            x = np.arange(1, len(PPV_additional) + 1)
            color = colors[i % len(colors)]  # Cycle through colors if necessary
            plt.semilogx(x, PPV_additional, ".-", label=labels[i + 1], color=color)
    else:
        # If only one PPV sequence, plot it without labels
        PPV_single = PPVs[0]
        x0 = np.arange(1, len(PPV_single) + 1)
        plt.semilogx(x0, PPV_single, ".-", markersize=1)
    
    # Set font sizes for ticks
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    
    # Add legend if multiple PPVs are plotted
    if l > 1:
        plt.legend(fontsize=fs)
    
    # Label axes with specified font size
    plt.xlabel("Number of Predictions", fontsize=fs)
    plt.ylabel("PPV", fontsize=fs)
    
    # Set plot title with specified font size
    plt.title(figtitle, fontsize=fs)
    plt.legend()
    
    # Optionally, add grid for better readability
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.savefig('./figs/PPV_{figtitle}.pdf'.format(figtitle=figtitle))
    # Display the plot
    #plt.show()



def contact_plot(score, filestruct, L, figurename,
                ticks=None, min_separation=0, cutoff=8.0, N="2L"):
    """
    Plot the contact map with true and false predictions.
    
    Parameters:
    - score: list of lists or tuples, where each element has at least 3 elements (i, j, ...)
    - filestruct: path to structure file or data, used by AttentionDCA.compute_residue_pair_dist
    - L: number of predictions to consider
    - figurename: string for the plot title
    - ticks: list or array for axis ticks (optional)
    - min_separation: minimum separation between residues (optional, default 0)
    - cutoff: maximum allowed distance (optional, default 8.0)
    - N: label for the plot, default "2L"
    """
    # Close all existing figures
    plt.close('all')
    # Create a new figure with default size
    plt.figure(figsize=(8, 6))
    
    # Compute residue pair distances
    dist = compute_residue_pair_dist(filestruct)  # dict with keys=(i,j), values=distance
    
    # Compute referencescore
    referencescore = compute_referencescore(score, dist, cutoff=cutoff, mindist=min_separation)
    # roc = map(x -> x[4], AttentionDCA.compute_referencescore(...))
    # Assuming referencescore is a list of lists or tuples with at least 4 elements
    roc = [x[3] for x in referencescore]  # x[4] in Julia corresponds to x[3] in Python (0-based indexing)
    
    # Compute precision
    if L <= len(roc):
        precision = round(roc[L-1], 2)
    else:
        precision = round(roc[-1], 2)  # If L exceeds, take last element
    
    # Filter dist by removing contacts with j - i <= min_separation, or distance > cutoff, or distance ==0
    keys_to_delete = [key for key, value in dist.items()
                      if (key[1] - key[0] <= min_separation) or (value > cutoff) or (value == 0)]
    
    for key in keys_to_delete:
        del dist[key]
    
    # Collect predicted_contacts: first L contacts from score where j - i > min_separation
    predicted_contacts = []
    i = 0  # Python indices start at 0
    while len(predicted_contacts) < L and i < len(score):
        contact = score[i]
        # Assuming contact[0] = i, contact[1] = j
        if contact[1] - contact[0] > min_separation:
            predicted_contacts.append( (contact[0], contact[1]) )
        i += 1
    
    # Split predicted_contacts into true_contacts and false_contacts
    true_contacts = []
    false_contacts = []
    for contact in predicted_contacts:
        if contact in dist:
            true_contacts.append(contact)
            del dist[contact]  # Remove from dist as it's a true contact
        else:
            false_contacts.append(contact)
    
    # Convert to NumPy arrays for plotting
    if true_contacts:
        true_contacts_np = np.array(true_contacts).T  # Shape: (2, M)
    else:
        true_contacts_np = np.empty((2, 0))
    
    if false_contacts:
        false_contacts_np = np.array(false_contacts).T  # Shape: (2, M)
    else:
        false_contacts_np = np.empty((2, 0))
    
    if dist:
        dist_np = np.array(list(dist.keys())).T  # Shape: (2, M)
    else:
        dist_np = np.empty((2, 0))
    
    # Plot dist contacts in black circles
    if dist_np.size != 0:
        plt.plot(dist_np[0], dist_np[1], 'ko', alpha=0.1, markersize=2, markeredgewidth=2)
        plt.plot(dist_np[1], dist_np[0], 'ko', alpha=0.1, markersize=2, markeredgewidth=2)
    
    # Plot true_contacts in blue circles
    if true_contacts_np.size != 0:
        plt.plot(true_contacts_np[0], true_contacts_np[1], 'bo', alpha=0.5, markersize=2, markeredgewidth=2, label='True Contacts')
        plt.plot(true_contacts_np[1], true_contacts_np[0], 'bo', alpha=0.5, markersize=2, markeredgewidth=2)
    
    # Plot false_contacts in red circles
    if false_contacts_np.size != 0:
        plt.plot(false_contacts_np[1], false_contacts_np[0], 'ro', alpha=0.5, markersize=2, markeredgewidth=2, label='False Contacts')
        plt.plot(false_contacts_np[0], false_contacts_np[1], 'ro', alpha=0.5, markersize=2, markeredgewidth=2)
    
    # Set ticks if provided
    if ticks is not None:
        plt.xticks(ticks, fontsize=10)
        plt.yticks(ticks, fontsize=10)
    
    # Set axis limits
    if dist_np.size != 0:
        max_dist = dist_np.max()
    else:
        max_dist = 1  # Default value if no distances exist
    plt.xlim(1, max_dist)
    plt.ylim(max_dist, 1)
    
    # Adjust layout for better spacing
    plt.tight_layout(pad=1.0)
    # Scale axes equally
    plt.axis('scaled')
    
    # Set plot title with precision
    plt.title(f"{figurename} Contact Map, P@{N}: {precision}", fontsize=20)
    
    # Add legend if there are true or false contacts
    if true_contacts_np.size != 0 or false_contacts_np.size != 0:
        plt.legend(fontsize=20)
    
    # Display the plot
    plt.show()

    return false_contacts


def contact_plot_from_map(score, distancemap_file, Ntop, figurename,
                          min_separation=5, cutoff=10.0, size=1):
    """
    Plot the contact map with true and false predictions.
    
    Parameters:
    - score: list of lists or tuples, where each element has at least 3 elements (i, j, ...)
             i and j are indices of positions, start from 1 #TODO: change
    - distancemap_file: path to distance map file (LxL matrix with distances between each pair of residues)
    - Ntop: number of predictions to consider
    - figurename: string for the plot title
    - min_separation: minimum separation between residues (optional, default 0)
    - cutoff: maximum allowed distance (optional, default 8.0)
    """
    score=np.array(score)
    distance_map = np.loadtxt(distancemap_file)

    # Filter distance_map by removing contacts with j - i <= min_separation
    np.fill_diagonal(distance_map, 100)
    for i in range(min_separation):
        np.fill_diagonal(distance_map[:, i+1:], 100)
        np.fill_diagonal(distance_map[i+1:], 100)


    PPV = compute_PPV_from_map(score, distancemap_file, min_separation=min_separation, cutoff=cutoff, nb_pred=Ntop+1) #TODO: rewrite func
    # Compute precision
    if Ntop <= len(PPV):
        precision = round(PPV[Ntop-1], 2)
    else:
        print("Ntop > len(PPV)!!!")
        precision = round(PPV[-1], 2)  # If L exceeds, take last element
    
    #Collect predicted contacts scores
    score=score[:Ntop]

    pred_contacts=np.zeros(score.shape, dtype=int)
    pred_contacts[:,0]=score[:,0]-1 # Index i, needs to start from 0 #TODO: change changing score to 0-based
    pred_contacts[:,1]=score[:,1]-1 # Index j

    # Label predicted contacts as true or false
    for i in range(len(pred_contacts)):
        if distance_map[pred_contacts[i][0], pred_contacts[i][1]] > cutoff:
            pred_contacts[i][2] = 0
        else:
            pred_contacts[i][2] = 1

    
    # Get true contacts, to make plotting easier
    contacts = np.where(distance_map <= cutoff)

    # Close all existing figures
    plt.close('all')
    plt.figure(figsize=(10,9))
    for i in range(len(contacts)):
        plt.scatter(contacts[0], contacts[1], color='grey', s=size, alpha=0.1)
    for i in range(len(pred_contacts)):
        if pred_contacts[i][2] == 1:
            plt.scatter(pred_contacts[i][0], pred_contacts[i][1], color='blue', s=size)
            plt.scatter(pred_contacts[i][1], pred_contacts[i][0], color='blue', s=size)
        else:
            plt.scatter(pred_contacts[i][0], pred_contacts[i][1], color='red', s=size)
            plt.scatter(pred_contacts[i][1], pred_contacts[i][0], color='red', s=size)
    plt.title(f"{figurename} top {Ntop} contacts, P: {precision:.2f}")
    plt.grid()
    plt.savefig('./figs/contactmap_{figurename}_top_{Ntop}.pdf'.format(figurename=figurename, Ntop=Ntop))
    plt.show()