import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
font = {'size'   : 18}

matplotlib.rc('font', **font)


def compute_position_frequencies(sequence_array, W=None ,n_amino_acids=21):
    """
    Computes frequency of each amino acid at each position.

    Parameters:
        sequence_array (np.ndarray): shape (n_sequences, sequence_length),
                                     where each value is an integer representing an amino acid.
        n_amino_acids (int, optional): total number of possible amino acids.
                                       If None, inferred from max value in array.

    Returns:
        np.ndarray: shape (sequence_length, n_amino_acids), where each entry [i, j]
                    is the frequency of amino acid j at position i.
    """
    n_sequences, seq_length = sequence_array.shape
    if n_amino_acids is None:
        n_amino_acids = int(sequence_array.max()) + 1  # assumes 0-based encoding

    freq_matrix = np.zeros((seq_length, n_amino_acids), dtype=float)
    if W is None:
        for pos in range(seq_length):
            counts = np.bincount(sequence_array[:, pos], minlength=n_amino_acids)
            freq_matrix[pos] = counts / n_sequences  # convert to frequencies

        return freq_matrix
    else:
        for i in range(n_sequences):
            for j in range(seq_length):
                freq_matrix[j,sequence_array[i][j]]+=W[i]
        # # one-hot: (N, M, q)
        # Z=sequence_array.T
        # one_hot = np.eye(n_amino_acids, dtype=float)[Z]

        # # Multiply weights (broadcast W along N and q)
        # weighted = one_hot * W[None, :, None]  # (N, M, q)

        # # Sum over sequences (axis=1), result shape (N, q)
        # f = weighted.sum(axis=1).T  # (q, N)
        return freq_matrix



def compute_pairwise_frequencies(seq_array,W=None ,n_amino_acids=None):
    """
    Computes pairwise frequency f_{ij}(a, b) for all positions i, j and amino acids a, b.

    Parameters:
        seq_array (np.ndarray): shape (n_sequences, sequence_length),
                                with integers representing amino acids.
        n_amino_acids (int, optional): number of distinct amino acids (if not known, inferred).

    Returns:
        np.ndarray: shape (sequence_length, sequence_length, n_amino_acids, n_amino_acids)
                    with joint frequencies f_{ij}(a, b)
    """
    n_sequences, seq_len = seq_array.shape
    if n_amino_acids is None:
        n_amino_acids = int(seq_array.max()) + 1  # assumes 0-based encoding

    freq_matrix = np.zeros((seq_len, seq_len, n_amino_acids, n_amino_acids), dtype=float)
    if W is None:
        for s in seq_array:
            for i in range(seq_len):
                a = s[i]
                for j in range(seq_len):
                    b = s[j]
                    freq_matrix[i, j, a, b] += 1
        freq_matrix /= n_sequences
    else:
        count_seq=0
        for s in seq_array:
            for i in range(seq_len):
                a = s[i]
                for j in range(seq_len):
                    b = s[j]
                    freq_matrix[i, j, a, b] += W[count_seq]
            count_seq+=1


    # Normalize to get frequencies
    
    return freq_matrix

def compute_pariwise_corr_freq(seq_array,W=None,n_amino_acids=None):
    pairwise_freq=compute_pairwise_frequencies(seq_array,W=W,n_amino_acids=n_amino_acids)
    position_freq=compute_position_frequencies(seq_array,W=W, n_amino_acids=n_amino_acids)
    correlation_mat=pairwise_freq - np.einsum("ij,kl->ikjl",position_freq,position_freq)
    return correlation_mat


def plot_corr_aa_pos(gen_seq, ref_seq, W=None, save_path=None, filename=None, beta=None, beta_PCA=None):
    # Compute pairwise correlation frequencies
    freq_mat_gen = compute_pariwise_corr_freq(gen_seq, W=None)
    freq_mat_data_train = compute_pariwise_corr_freq(ref_seq, W=W)

    # Flatten
    freq_mat_gen = np.reshape(freq_mat_gen, -1)
    freq_mat_data_train = np.reshape(freq_mat_data_train, -1)

    # Scatter plot
    if beta_PCA is not None:
        plt.scatter(freq_mat_data_train, freq_mat_gen, label=fr'$\beta={beta}, \beta_P={beta_PCA}$', alpha=0.6)
    else:
        plt.scatter(freq_mat_data_train, freq_mat_gen, label=fr'$\beta={beta}$', alpha=0.6)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(freq_mat_data_train, freq_mat_gen)
    print(f"Linear fit slope = {slope:.4f}, intercept = {intercept:.4f}, R² = {r_value**2:.4f}")

    # Plot regression line
    x_vals = np.linspace(min(freq_mat_data_train), max(freq_mat_data_train), 200)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=2, label=f"Fit slope={slope:.3f}")

    # Diagonal reference line
    mmax = max(max(freq_mat_data_train), max(freq_mat_gen))
    plt.plot([0, mmax], [0, mmax], color='gray', linestyle='--')

    # Labels and grid
    plt.xlabel(r"Reference $C_{ij,ab}$")
    plt.ylabel(r"Generated $C_{ij,ab}$")
    plt.grid(True)
    plt.legend()

    # Save if required
    if save_path:
        save_name = save_path + f'/freqs_{filename}.png'
        plt.savefig(save_name)

    plt.show()

def plot_frequencies_aa_pos(gen_seq, ref_seq, W=None, save_path=None, filename=None, beta=None, beta_PCA=None):
    # Compute amino acid position frequencies
    freq_mat_gen = compute_position_frequencies(gen_seq, W=None)
    freq_mat_data_train = compute_position_frequencies(ref_seq, W=W)

    # Flatten
    freq_mat_gen = np.reshape(freq_mat_gen, -1)
    freq_mat_data_train = np.reshape(freq_mat_data_train, -1)

    # Scatter plot
    if beta_PCA is not None:
        plt.scatter(freq_mat_data_train, freq_mat_gen, label=fr'$\beta={beta}, \beta_P={beta_PCA}$', alpha=0.6)
    else:
        plt.scatter(freq_mat_data_train, freq_mat_gen, label=fr'$\beta={beta}$', alpha=0.6)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(freq_mat_data_train, freq_mat_gen)
    print(f"Linear fit slope = {slope:.4f}, intercept = {intercept:.4f}, R² = {r_value**2:.4f}")

    # Regression line
    x_vals = np.linspace(min(freq_mat_data_train), max(freq_mat_data_train), 200)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=2, label=f"Fit slope={slope:.3f}")

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Labels and grid
    plt.xlabel("Reference Amino Acid Frequency")
    plt.ylabel("Generated Amino Acid Frequency")
    plt.grid(True)
    plt.legend()

    # Save if needed
    if save_path:
        save_name = save_path + f'/freqs_{filename}.png'
        plt.savefig(save_name)

    plt.show()
