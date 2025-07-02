from seq_utils import letters_to_nums, sequences_from_fasta
#from plm_PCA import one_hot_seq_batch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

#################################################################
#        ----------------FUNCTIONS----------------

def is_duplicate_sequence(gen_seq, ref_seqs):
    """
    Check if a single sequence exists in a reference set.
        sequence (np.array): 1D array of one gen sequence
        reference_set (np.array): array of Train/Test sequences)
    Returns:
        bool: True if sequence exists in the reference set, False otherwise.
    """
    return any(np.array_equal(gen_seq, ref_seq) for ref_seq in ref_seqs)


def check_random_duplicates(gen_sequences, reference_set, N_seq=100):
    """
    Randomly sample N_seq sequences from gen_sequences and check for duplicates in reference_set.
        gen_sequences (np.array)
        reference_set (np.array): Train/Test sequences
        N_seq (int): Number of random sequences to check.
    Returns:
        List of indices of duplicate sequences.
    """
    total_gen = gen_sequences.shape[0]
    sampled_indices = random.sample(range(total_gen), N_seq)
    duplicates = []
    tot_dup = 0
    for idx in tqdm(sampled_indices):
        seq = gen_sequences[idx]
        if is_duplicate_sequence(seq, reference_set):
            duplicates.append(idx)
            tot_dup += 1
    print(f"Total duplicates found: {tot_dup} out of {N_seq} sampled sequences.")
    return duplicates

#################################################################
#           ----------------MAIN-----------------

# Define the file path
filename = 'generated_sequences_40000'
cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main'
output_file = cwd + f'/CODE/AttentionDCA_python/src/PLM/generated_sequences/{filename}.npy'

family = 'jdoms_bacteria_train2'
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'

# Get the raw letter sequences from the FASTA file
folder_name = "/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main/CODE/AttentionDCA_python/src/my_saved_data"
os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, "plm_generated_V21_gap_seqs_jdom_40000_exp_pos_init_mod.txt")
seq_aa=np.loadtxt(file_path,dtype=int)
gen_sequences = seq_aa
#gen_sequences = np.load(output_file)
train_sequences = sequences_from_fasta(filename)
# Convert to numeric sequences
train_sequences_num = np.array([letters_to_nums(seq) for seq in train_sequences])

# Test sequences
family = 'jdoms_bacteria_test2'
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'
# Initialize a list to store the sequences
test_sequences = sequences_from_fasta(filename)
test_sequences_num = np.array([letters_to_nums(seq) for seq in test_sequences])

print(check_random_duplicates(gen_sequences, train_sequences_num, N_seq=1000))
print(check_random_duplicates(gen_sequences, test_sequences_num, N_seq=1000))