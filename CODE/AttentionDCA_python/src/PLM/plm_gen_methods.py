from plm_model import SequencePLM
from seq_utils import nums_to_letters

import os
import numpy as np
from tqdm import tqdm

def generate_plm(J,N_seqs=40000, init_sequence=None,beta=1,nb_PCA_comp=0,PCA_comp_list=np.array([]),J_PCA=None,beta_PCA=1):
    """
    Generate N_seqs new sequences using PLM (random initialization by default)
    """
    gen_sequences = []
    seq = SequencePLM(J, init_sequence,beta=beta,nb_PCA_comp=nb_PCA_comp,PCA_component_list=PCA_comp_list,J_tens_PCA=J_PCA,beta_PCA=beta_PCA)
    for _ in tqdm(range(N_seqs)):
        site = np.random.randint(seq.L) # Random site from 0 to L-1
        seq.draw_aa(site)
        gen_sequences.append(seq.sequence.copy())
    gen_sequences = np.array(gen_sequences)
    return gen_sequences


def generate_plm_alter(J, N_seqs = 10000, N_iters=1000 , init_sequence=None,beta=1):
    """
    Generate N_seqs with N_iters draws for each sequence.
    """
    gen_sequences = []
    seq = SequencePLM(J, init_sequence,beta=beta)
    for _ in tqdm(range(N_seqs)):
        for _ in range(N_iters):
            site = np.random.randint(seq.L)  # Random site from 0 to L-1
            seq.draw_aa(site)
        gen_sequences.append(seq.sequence.copy())
    gen_sequences = np.array(gen_sequences)
    return gen_sequences

def generate_plm_n_save(save_dir, save_name, J, N_seqs=10000, init_sequence=None,beta=1,nb_PCA_comp=0,PCA_comp_list=np.array([]),J_PCA=None,beta_PCA=1):
    """
    Generates a set of sequences using the PLM and saves them both as a numpy file and a text file containing the corresponding letter sequences.
    Saves:
    - A `.npy` file containing the generated sequences in numerical format.
    - A `.txt` file containing the generated sequences in letter format.
    """
    gen_sequences = generate_plm(J, N_seqs, init_sequence,beta=beta,nb_PCA_comp=nb_PCA_comp,PCA_comp_list=PCA_comp_list,J_PCA=J_PCA,beta_PCA=beta_PCA)
    gen_sequences_letters = [nums_to_letters(sequence,nb_PCA_comp) for sequence in gen_sequences]
    
    print(f"Generated sequences (letters): {gen_sequences_letters[:5]}")  # Show first 5 sequences
    
    gen_sequences = np.array(gen_sequences)
    if nb_PCA_comp!=0:
        save_dir=save_dir+"PCA_comp"
        save_name=save_name+"PCA_comp"
        for i in gen_sequences[0,len(gen_sequences)-nb_PCA_comp:]:
            save_dir+=str(i)+"_"  
    # Check if the directory exists, create it if not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the sequences in numerical format as a .npy file
    np.save(f"{save_dir}/{save_name}.npy", gen_sequences)
    
    # Save the sequences in letter format as a .txt file (each sequence on a new line)
    with open(f"{save_dir}/{save_name}.txt", "w") as f:
        for sequence in gen_sequences_letters:
            f.write(f"{sequence}\n")

    print(f"Generated sequences saved to {save_dir}")