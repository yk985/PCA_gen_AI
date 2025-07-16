import numpy as np
import torch
import random
import re
from pathlib import Path



letter_to_num = {
    'A': 0,  'B': 20, 'C': 1,  'D': 2,  'E': 3,
    'F': 4,  'G': 5,  'H': 6,  'I': 7,  'J': 20,
    'K': 8,  'L': 9, 'M': 10, 'N': 11, 'O': 20,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'U': 20, 'V': 17, 'W': 18, 'X': 20, 'Y': 19,
    '-': 20  # Gap symbol
}

num_to_letter  = {v: k for k, v in letter_to_num.items()}

################################################################

def read_tensor_from_txt(filename):
    """
        Usual method to get Q, K, V tensors from text files
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Read the dimensions from the first line
    dims = list(map(int, lines[0].strip().split()))
    
    tensor_data = []
    current_slice = []
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Slice"):
            if current_slice:
                tensor_data.append(current_slice)
                current_slice = []
        elif line:
            current_slice.append(list(map(float, line.split(','))))
    if current_slice:
        tensor_data.append(current_slice)

    tensor = torch.tensor(tensor_data).view(*dims)
    return tensor

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sequences_from_fasta(file_path):
    "Reads fasta file and returns list of letter sequences"
    sequences = []
    with open(file_path, "r") as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Ignore headers
                if sequence:  # If there's a previous sequence, add it to the list
                    sequences.append(sequence)
                sequence = ''  # Reset sequence for the next one
            else:
                sequence += line  # Add the line to the current sequence
        if sequence:  # Append the last sequence
            sequences.append(sequence)
    return sequences

################################################################
# Sequence manipulations

def modify_seq(seq,ratio,aa_list=np.arange(21),nb_PCA_comp=0):
    """
    Randomly modifies a sequence by changing a chosen ratio of its elements to random amino acids.
    """
    L = len(seq)-nb_PCA_comp
    seq_func = seq.copy()
    nb_change = int(L*ratio)
    ind_change = np.random.choice(np.arange(L), nb_change)
    for i in range(nb_change):
        seq_func[ind_change[i]] = np.random.choice(aa_list)
    return seq_func

def one_hot_seq_batch(seqs, max_pot=21):
    def one_hot_aa(aa):
        zeros = np.zeros(max_pot)
        zeros[aa] = 1  # assumes aa in 0-20
        return zeros
    return np.array([[one_hot_aa(aa) for aa in seq] for seq in seqs])

def invert_dict(d):
    inverted = {}
    for key, value in d.items():
        if value in inverted:
            inverted[value].append(key)
        else:
            inverted[value] = [key]
    return inverted


### choose which to keep - i thought if we define lettter_to_num as global dictionnary no need to call inside functions
def numbers_to_letters(numbers, inverted_dict):
    result = ''
    for number in numbers:
        if number in inverted_dict:
            letters = inverted_dict[number]
            result += random.choice(letters)
        else:
            raise ValueError(f"Number {number} not found in the inverted dictionary.")
    return result

def seq_num_to_letters(file_path,dictionary):
    inv_dict=invert_dict(dictionary)
    seq_array=np.loadtxt(file_path).astype(np.int64)
    letter_seq_array=[]
    for seq in seq_array:
        letter_seq_array.append(numbers_to_letters(seq,inv_dict))
    output_path=file_path.replace(".txt",'')+"output.txt"
    np.savetxt(output_path, np.array(letter_seq_array), fmt='%s')

################################################################


def letters_to_nums(sequence):
    "receives AA sequence (format ABC... and reutrns [1 2 3 ...])"
    return np.array([letter_to_num.get(aa, 20) for aa in sequence])

def nums_to_letters(sequence,nb_PCA_comp=0):
    """Receives a sequence of integers (e.g., [1, 2, 4]) and returns a string of corresponding amino acid letters."""
    num_to_letter = {v: k for k, v in letter_to_num.items()}
    return ''.join([num_to_letter.get(num, 'X') for num in sequence[:len(sequence)-nb_PCA_comp]])

###############################################################

def extract_beta_beta_PCA(filename):
    stem = Path(filename).stem      # -> 'gill_gen_seqs_randinit_Ns8000_b_1_b_PCA12_PCA_comp_24_20PCA_comp'

    # capture the digits after  b_   and after  b_PCA
    m = re.search(r"b_(\d+)_b_PCA(\d+)", stem)
    if m:
        after_b      = int(m.group(1))   # 1
        after_b_PCA  = int(m.group(2))   # 12
        return after_b, after_b_PCA
    else:
        raise ValueError("Expected pattern not found")


