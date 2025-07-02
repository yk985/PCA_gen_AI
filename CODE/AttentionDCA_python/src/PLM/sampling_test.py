import numpy as np
import matplotlib.pyplot as plt
from PCA_func import plot_pca_of_sequences
from hamming_dist import *


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

def compare_seq_arrays(gen_seq,test_data,initial_seq=None,pca_graph_restrict=True):
    #PCA plot
    plot_pca_of_sequences(gen_seq,comparison_data=test_data,pca_graph_restrict=pca_graph_restrict)
    # AA and position frequencies
    freq_mat_gen=compute_position_frequencies(gen_seq)
    freq_mat_data_train=compute_position_frequencies(test_data)
    freq_mat_gen=np.reshape(freq_mat_gen,-1)
    freq_mat_data_train=np.reshape(freq_mat_data_train,-1)
    plt.scatter(freq_mat_data_train,freq_mat_gen)
    plt.show()
    if initial_seq is None:
        initial_seq=gen_seq[0]
    ham_list=hamming_dist_oneseq_to_batch(initial_seq,gen_seq)
    plt.plot(ham_list)
    plt.xlabel("Iterations")
    plt.ylabel("hamming distance with initial seq")
    plt.show()
    ham_correlation=energy_corr_array(ham_list,5000)
    plt.plot(ham_correlation)
    plt.xlabel("Tau")
    plt.ylabel("Correlation")
    plt.show()
    ham_gen_test=vectorized_hamming_distance(gen_seq,test_data)
    print("Average Hamming distance (generated/test data):", np.mean(ham_gen_test))
    print("STD Hamming distance:", np.std(ham_gen_test))


    