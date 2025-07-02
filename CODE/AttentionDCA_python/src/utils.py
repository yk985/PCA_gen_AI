import numpy as np
import gzip
from sklearn.metrics import pairwise_distances
import gzip
from tqdm import tqdm




# Mapping of amino acids to integers
letter_to_num_ale = {
    'A': 1,  'B': 21, 'C': 2,  'D': 3,  'E': 4,
    'F': 5,  'G': 6,  'H': 7,  'I': 8,  'J': 21,
    'K': 9,  'L': 10, 'M': 11, 'N': 12, 'O': 21,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
    'U': 21, 'V': 18, 'W': 19, 'X': 21, 'Y': 20,
    '-': 21  # Gap symbol
}

letter_to_num = {
    'A': 0,  'B': 20, 'C': 1,  'D': 2,  'E': 3,
    'F': 4,  'G': 5,  'H': 6,  'I': 7,  'J': 20,
    'K': 8,  'L': 9, 'M': 10, 'N': 11, 'O': 20,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'U': 20, 'V': 17, 'W': 18, 'X': 20, 'Y': 19,
    '-': 20  # Gap symbol
}

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PLM.seq_utils import one_hot_seq_batch

def get_sequences_pca_coords(sequences_train, max_pot=21):
    """
    Compute 2D PCA coordinates from training sequences using one-hot + scaling.
    """
    one_hot_train = one_hot_seq_batch(sequences_train, max_pot=max_pot)  # shape (num_seq, seq_len, max_pot)
    train_flat = one_hot_train.reshape(one_hot_train.shape[0], -1)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_scaled)
    return train_pca

def get_PCA_grid_coords(coords_2d, N, plot=False, highlight_index=None):
    """
    Normalize and discretize 2D PCA coordinates into an NxN integer grid.
    """
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    norm_coords = (coords_2d - coords_min) / (coords_max - coords_min + 1e-8)
    grid_coords = (norm_coords * (N - 1)).astype(int)
    
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(norm_coords[:, 0], norm_coords[:, 1], c='gray', s=40, label="All sequences")
        for i in range(N + 1):
            plt.axhline(i / (N - 1), color='lightgray', linewidth=0.5)
            plt.axvline(i / (N - 1), color='lightgray', linewidth=0.5)
        if highlight_index is not None:
            x, y = norm_coords[highlight_index]
            gx, gy = grid_coords[highlight_index]
            plt.scatter(x, y, color='red', s=100, label=f"Sequence {highlight_index}")
            plt.text(x + 0.01, y + 0.01, f"Grid: ({gx},{gy})", color='red', fontsize=10)
        plt.title(f"PCA Projection with {N}x{N} Grid")
        plt.xlabel("PCA 1 (normalized)")
        plt.ylabel("PCA 2 (normalized)")
        plt.legend()
        plt.grid(False)
        plt.show()
    return grid_coords

def add_PCA_coords(seqs, N, max_pot=21,  plot= False, highlight_index=None):
    """ Parameters: 
        - seqs: numpy array of training sequences (length L)
        - N: int, discretization of PCA into 2D grid NxN
        Returns: 
        - seqs_PCA: numpay array of sequences with PCA coordinates added (Length: L+2)"""
    # 1) Compute 2D PCA coordinates from sequences (using your PCA function)
    pca_coords = get_sequences_pca_coords(seqs, max_pot=max_pot)  # shape (num_seq, 2)

    # 2) Discretize PCA coords into NxN grid
    grid_coords = get_PCA_grid_coords(pca_coords, N, plot= plot, highlight_index=highlight_index)  # shape (num_seq, 2)

    # 3) Convert seqs to array if needed and ensure proper shape for hstack
    seqs_array = np.array(seqs)
    if seqs_array.ndim == 1:  # 1D list of sequences (e.g., strings)
        raise ValueError("seqs should be converted to numeric array before adding PCA coords.")

    # 4) Stack original features + grid coordinates
    seqs_pca = np.hstack((seqs_array, grid_coords))

    # 5) Check shapes
    expected_cols = seqs_array.shape[1] + 2
    if seqs_pca.shape[1] != expected_cols:
        raise ValueError(f"Expected shape ({seqs_array.shape[0]}, {expected_cols}), got {seqs_pca.shape}")

    return seqs_pca

def open_fasta(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    else:
        return open(filename, 'r')

def read_fasta_alignment(filename, max_gap_fraction,  verbose=True, return_filtered=False):
    """
    Reads a FASTA alignment file and returns the alignment matrix Z.

    Parameters:
    - filename: str, path to the FASTA file.
    - max_gap_fraction: float, maximum allowed fraction of gaps in a sequence.
    - max_sequences: int or None, maximum number of sequences to read. If None, read all sequences.
    - verbose: bool, whether to print status messages.

    Returns:
    - Z: numpy array of shape (sequence_length, num_sequences), the alignment matrix.
    """
    max_sequences= None
    max_gap_fraction = float(max_gap_fraction)
    sequences = []
    names = []
    with open_fasta(filename) as f:
        seq_name = None
        seq_lines = []
        seq_count = 0

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq_name is not None:
                    sequences.append((seq_name, ''.join(seq_lines)))
                    seq_count += 1
                    if max_sequences is not None and seq_count >= max_sequences:
                        if verbose:
                            print(f"Reached maximum number of sequences: {max_sequences}")
                        break
                seq_name = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.upper())

        if seq_name is not None and (max_sequences is None or seq_count < max_sequences):
            sequences.append((seq_name, ''.join(seq_lines)))
            seq_count += 1

    if not sequences:
        raise ValueError("No sequences found in the file.")

    if verbose:
        print(f"Total sequences read: {len(sequences)}")

    # First pass: Determine positions to include based on the first sequence
    first_seq = sequences[0][1]
    indices = []
    fseqlen = 0
    for i, c in enumerate(first_seq):
        if c != '.' and not c.islower():
            fseqlen += 1
            indices.append(i)

    # Filter sequences based on max_gap_fraction
    filtered_sequences = []
    for name, seq in sequences:
        if len(seq) != len(first_seq):
            raise ValueError("Sequences are not aligned; they have different lengths.")
        ngaps = 0
        for idx in indices:
            c = seq[idx]
            #if c=='-':
            if letter_to_num[c] == 20: #changed by youssef to remove all amino acids that are unknown 
                ngaps += 1
        gap_fraction = ngaps / fseqlen
        if gap_fraction <= max_gap_fraction:
            filtered_sequences.append((name, seq))
        else:
            if verbose:
                print(f"Sequence {name} excluded due to high gap fraction ({gap_fraction:.2f}).")

    if not filtered_sequences:
        raise ValueError(f"Out of {len(sequences)} sequences, none passed the filter (max_gap_fraction={max_gap_fraction})")

    if verbose:
        print(f"Sequences after filtering: {len(filtered_sequences)}")

    # Second pass: Construct the alignment matrix Z
    num_seqs = len(filtered_sequences)
    sequence_length = fseqlen
    Z = np.empty((sequence_length, num_seqs), dtype=np.int8)
    for seq_idx, (name, seq) in enumerate(filtered_sequences):
        for pos_idx, idx in enumerate(indices):
            c = seq[idx]
            num = letter_to_num.get(c, 21)
            Z[pos_idx, seq_idx] = num
    if return_filtered:
        return Z, filtered_sequences
    else:
        return Z



def remove_duplicate_sequences(Z, verbose=True):
    """
    Removes duplicate sequences from the alignment matrix Z.

    Parameters:
    - Z: numpy array of shape (sequence_length, num_sequences)
    - verbose: bool, whether to print status messages.

    Returns:
    - newZ: numpy array with duplicate sequences removed.
    - unique_indices: list of indices of unique sequences.
    """
    M = Z.shape[1]
    if verbose:
        print("Removing duplicate sequences...")

    # Use a dictionary to track unique sequences
    sequence_dict = {}
    unique_indices = []
    for i in range(M):
        seq_bytes = Z[:, i].tobytes()
        if seq_bytes not in sequence_dict:
            sequence_dict[seq_bytes] = i
            unique_indices.append(i)

    newZ = Z[:, unique_indices]
    if verbose:
        print(f"Done: {M} -> {len(unique_indices)} sequences after removing duplicates.")
    return newZ, unique_indices
    




def compute_theta(Z, verbose=True):
    """
    Compute the theta value based on the mean fraction of identical positions between sequences.

    Parameters:
    - Z: numpy array of shape (N, M), where N is the sequence length, M is the number of sequences.
    - verbose: bool, whether to print intermediate information.

    Returns:
    - theta: float
    """
    N, M = Z.shape
    Z = Z.T  # Transpose to shape (M, N), where each row is a sequence

    # Compute pairwise Hamming distances (normalized by sequence length)
    if verbose:
        print("Computing pairwise Hamming distances...")
    hamming_distances = pairwise_distances(Z, metric='hamming')  # Shape: (M, M)

    # Fraction of identical positions is 1 - Hamming distance
    fracid_matrix = 1 - hamming_distances

    # Get upper triangle indices (excluding the diagonal)
    iu = np.triu_indices(M, k=1)

    # Extract fractions for upper triangle pairs
    fracid = fracid_matrix[iu]  # Shape: (num_pairs,)

    # Compute mean fraction of identical positions
    meanfracid = np.mean(fracid)

    if verbose:
        print(f"Mean fraction of identical positions: {meanfracid}")

    # Compute theta
    theta = min(0.5, 0.1216 / meanfracid)  # 0.38 * 0.32 = 0.1216

    if verbose:
        print(f"Computed theta: {theta}")

    return theta


def compute_theta_sampled(Z, num_samples=100000, verbose=True):
    """
    Compute the theta value based on the mean fraction of identical positions between sequences,
    using a random subset of pairs if necessary.

    Parameters:
    - Z: numpy array of shape (N, M), where N is the sequence length, M is the number of sequences.
    - num_samples: int, number of pairs to sample if M is large.
    - verbose: bool

    Returns:
    - theta: float
    """
    N, M = Z.shape
    Z = Z.T  # Shape: (M, N)

    total_pairs = M * (M - 1) // 2

    if total_pairs <= num_samples:
        # Use all pairs
        if verbose:
            print("Using all pairs.")
        return compute_theta(Z.T, verbose=verbose)
    else:
        if verbose:
            print(f"Sampling {num_samples} pairs out of {total_pairs} total pairs.")

        # Generate random pairs (i, j) with i < j
        sampled_i = np.random.randint(0, M, size=num_samples * 2)
        sampled_j = np.random.randint(0, M, size=num_samples * 2)
        mask = sampled_i < sampled_j
        sampled_i = sampled_i[mask][:num_samples]
        sampled_j = sampled_j[mask][:num_samples]

        # Get the sequences
        Z_i = Z[sampled_i]
        Z_j = Z[sampled_j]

        # Compute fraction of identical positions
        fracid = np.mean(Z_i == Z_j, axis=1)  # Shape: (num_samples,)

        # Compute meanfracid
        meanfracid = np.mean(fracid)

        if verbose:
            print(f"Mean fraction of identical positions (sampled): {meanfracid}")

        # Compute theta
        theta = min(0.5, 0.1216 / meanfracid)

        if verbose:
            print(f"Computed theta: {theta}")

        return theta

def compute_weights_large(Z, theta, verbose=True):
    """
    Compute sequence weights based on sequence similarity for large datasets.

    Parameters:
    - Z: numpy array of shape (N, M), where N is sequence length and M is the number of sequences.
         Each column represents a sequence.
    - theta: float, distance threshold between 0 and 1.
    - verbose: bool, whether to print intermediate information.

    Returns:
    - W: numpy array of shape (M,), the weight of each sequence.
    - Meff: float, the sum of weights (effective number of sequences).
    """
    N, M = Z.shape


    if theta == 'auto':
        theta = compute_theta_sampled(Z)

    if theta == 0:
        W = np.ones(M)
        Meff = M
        if verbose:
            print(f"M = {M} N = {N} Meff = {Meff}")
        return W, Meff

    thresh = np.floor(theta * N)

    if verbose:
        print(f"θ = {theta} threshold = {thresh}")



    W = np.zeros(M, dtype=np.float64)

    # Convert Z to a suitable data type to save memory (e.g., np.uint8)
    Z = Z.astype(np.uint8)

    # Define batch size (adjust based on your system's memory)
    batch_size = 1000

    # Iterate over sequences
    for i in tqdm(range(M)):
        if (i+1) % 1000 == 0 and verbose:
            print(f"Processing sequence {i+1}/{M}")

        Z_i = Z[:, i]  # Shape: (N,)

        # Compare Z_i with sequences j > i
        for j_start in range(i + 1, M, batch_size):
            j_end = min(j_start + batch_size, M)
            Z_batch = Z[:, j_start:j_end]  # Shape: (N, batch_size)

            # Compute differences
            differences = Z_i[:, np.newaxis] != Z_batch  # Shape: (N, batch_size)

            # Compute Hamming distances
            dists = np.sum(differences, axis=0)  # Shape: (batch_size,)

            # Identify similar sequences
            similar_indices = np.where(dists < thresh)[0]

            # Increment counts
            W[i] += len(similar_indices)
            W[j_start + similar_indices] += 1

    # Compute weights
    W = 1 / (1 + W)

    # Compute effective number of sequences
    Meff = np.sum(W)

    if verbose:
        print(f"M = {M} N = {N} Meff = {Meff}")

    return W, Meff

def compute_weights(Z, theta, verbose=True):
    """
    Compute sequence weights based on sequence similarity.

    Parameters:
    - Z: numpy array of shape (N, M), where N is sequence length and M is the number of sequences.
         Each column represents a sequence.
    - theta: float, distance threshold between 0 and 1.
    - verbose: bool, whether to print intermediate information.

    Returns:
    - W: numpy array of shape (M,), the weight of each sequence.
    - Meff: float, the sum of weights (effective number of sequences).
    """
    N, M = Z.shape

    if theta == 'auto':
        theta = compute_theta_sampled(Z)

    thresh = np.floor(theta * N)
    

    if verbose:
        print(f"θ = {theta} threshold = {thresh}")

    if theta == 0:
        W = np.ones(M)
        Meff = M
        if verbose:
            print(f"M = {M} N = {N} Meff = {Meff}")
        return W, Meff
    


    # Initialize weights
    W = np.zeros(M)

    # Compute pairwise Hamming distances
    # Expand dimensions for broadcasting
    Z_expanded_i = Z[:, :, np.newaxis]  # Shape: (N, M, 1)
    Z_expanded_j = Z[:, np.newaxis, :]  # Shape: (N, 1, M)

    # Compute differences
    differences = Z_expanded_i != Z_expanded_j  # Shape: (N, M, M)

    # Sum differences to get Hamming distances
    dist_matrix = np.sum(differences, axis=0)  # Shape: (M, M)

    # Get upper triangle indices (excluding diagonal)
    iu = np.triu_indices(M, k=1)

    # Extract distances for upper triangle
    dists = dist_matrix[iu]

    # Identify pairs with distance less than threshold
    similar_pairs = dists < thresh

    # Get indices of similar sequences
    indices_i = iu[0][similar_pairs]
    indices_j = iu[1][similar_pairs]

    # Increment counts for similar sequences
    np.add.at(W, indices_i, 1)
    np.add.at(W, indices_j, 1)

    # Compute weights
    W = 1 / (1 + W)

    # Compute effective number of sequences
    Meff = np.sum(W)

    if verbose:
        print(f"M = {M} N = {N} Meff = {Meff}")

    return W, Meff

def ReadFasta(filename, max_gap_fraction, theta='auto', remove_dups=True, verbose=True):


    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups:
        Z, _ = remove_duplicate_sequences(Z, verbose=verbose)
    N, M = Z.shape
    q = int(round(Z.max()))
    if q > 32:
        raise ValueError(f"Parameter q={q} is too big (max 31 is allowed)")
    theta = 'auto'
    W, Meff = compute_weights_large(Z, theta= theta, verbose=False)

    print(Meff)
    W /= Meff  # Normalize W
    Zint = Z.astype(int)
    return  W, Zint, N, M, q

def quickread(fastafile,max_gap_frac=0.9 ,moreinfo=False):
    W, Zint, N, M, _ = ReadFasta(fastafile, max_gap_frac, 'auto', True, verbose=False)
    if moreinfo:
        return W, Zint, N, M
    else:
        #create a folder and save Zint and W in it
        



        return Zint, W






















def load_matrix_3d_from_julia(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Read the dimensions from the first line
    H, d, N = map(int, lines[0].strip().split())
    
    matrices = []
    line_idx = 1  # Start reading from the second line
    
    for i in range(1, H + 1):
        # Read the "Slice i" line
        slice_header = lines[line_idx].strip()
        expected_header = f"Slice {i}"
        if slice_header != expected_header:
            raise ValueError(f"Expected '{expected_header}', got '{slice_header}'")
        line_idx += 1
        
        slice_data = []
        for j in range(d):
            row_str = lines[line_idx].strip()
            if not row_str:
                raise ValueError(f"Empty row encountered in slice {i}, row {j}")
            row = list(map(float, row_str.split(',')))
            if len(row) != N:
                raise ValueError(f"Expected {N} elements in row, got {len(row)}")
            slice_data.append(row)
            line_idx += 1
        
        matrices.append(slice_data)
        
        # Skip the empty line between slices
        if i < H:
            if lines[line_idx].strip() != "":
                raise ValueError(f"Expected empty line after Slice {i}, got '{lines[line_idx].strip()}'")
            line_idx += 1
    
    # Convert to NumPy array
    matrices_np = np.array(matrices)  # Shape: (H, d, N)
    return matrices_np