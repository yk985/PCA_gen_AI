import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
font = {'size'   : 18}

matplotlib.rc('font', **font)


def compute_position_frequencies(sequence_array, n_amino_acids=None):
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

    for pos in range(seq_length):
        counts = np.bincount(sequence_array[:, pos], minlength=n_amino_acids)
        freq_matrix[pos] = counts / n_sequences  # convert to frequencies

    return freq_matrix

def compute_pairwise_frequencies(seq_array, n_amino_acids=None):
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

    for s in seq_array:
        for i in range(seq_len):
            a = s[i]
            for j in range(seq_len):
                b = s[j]
                freq_matrix[i, j, a, b] += 1

    # Normalize to get frequencies
    freq_matrix /= n_sequences
    return freq_matrix

def compute_pariwise_corr_freq(seq_array,n_amino_acids=None):
    pairwise_freq=compute_pairwise_frequencies(seq_array,n_amino_acids=n_amino_acids)
    position_freq=compute_position_frequencies(seq_array, n_amino_acids=n_amino_acids)
    correlation_mat=pairwise_freq - np.einsum("ij,kl->ikjl",position_freq,position_freq)
    return correlation_mat


def plot_corr_aa_pos(gen_seq,ref_seq,save_path=None,filename=None,beta=None,beta_PCA=None):
    freq_mat_gen=compute_pariwise_corr_freq(gen_seq)
    freq_mat_data_train=compute_pariwise_corr_freq(ref_seq)
    freq_mat_gen=np.reshape(freq_mat_gen,-1)
    freq_mat_data_train=np.reshape(freq_mat_data_train,-1)
    if not beta_PCA is None:
        plt.scatter(freq_mat_data_train,freq_mat_gen,label=fr'$\beta={beta},\beta_P={beta_PCA}$')
    else:
        plt.scatter(freq_mat_data_train,freq_mat_gen,label=fr'$\beta={beta}$')
    mmax=max(max(freq_mat_data_train),max(freq_mat_gen))
    plt.plot([0,mmax],[0,mmax], color='gray', linestyle='--')
    plt.xlabel(r"Reference $C_{ij,ab}$")
    plt.ylabel(r"Generated $C_{ij,ab}$")
    plt.grid(True)
    plt.legend()
    # if beta:
    #     plt.plot([0], [0], ' ', label=fr'$\beta={beta}$')  # Dummy plot for legend
    #     plt.legend()

    if save_path:
        save_name = save_path + f'/freqs_{filename}.png'
        plt.savefig(save_name)
    plt.show()

def plot_frequencies_aa_pos(gen_seq,ref_seq,save_path=None,filename=None,beta=None,beta_PCA=None):
    freq_mat_gen=compute_position_frequencies(gen_seq)
    freq_mat_data_train=compute_position_frequencies(ref_seq)
    freq_mat_gen=np.reshape(freq_mat_gen,-1)
    freq_mat_data_train=np.reshape(freq_mat_data_train,-1)
    if not beta_PCA is None:
        plt.scatter(freq_mat_data_train,freq_mat_gen,label=fr'$\beta={beta},\beta_P={beta_PCA}$')
    else:
        plt.scatter(freq_mat_data_train,freq_mat_gen,label=fr'$\beta={beta}$')

    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.xlabel("Reference Amino Acid Frequency")
    plt.ylabel("Generated Amino Acid Frequency")
    plt.grid(True)
    plt.legend()
    # if beta:
    #     plt.plot([0], [0], ' ', label=fr'$\beta={beta}$')  # Dummy plot for legend
    #     plt.legend()

    if save_path:
        save_name = save_path + f'/freqs_{filename}.png'
        plt.savefig(save_name)
    plt.show()
