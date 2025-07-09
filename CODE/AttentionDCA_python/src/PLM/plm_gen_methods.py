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

def generate_PCA_coords(init_sequence, N_iter, J_PCA, nb_PCA_comp=2, beta_PCA=1, J=None, PCA_comp_list=None, beta=1):
    """
    Generate PCA coordinates for the sequences.
    """
    if J_PCA is None:
        return np.zeros(nb_PCA_comp)
    PCA_coords = []
    seq = SequencePLM(J, init_sequence,beta=beta,nb_PCA_comp=nb_PCA_comp,PCA_component_list=PCA_comp_list,J_tens_PCA=J_PCA,beta_PCA=beta_PCA)
    for _ in tqdm(range(N_iter)):
        PCA_coord = seq.update_PCA_coords()
        PCA_coords.append(PCA_coord)
    return PCA_coords
    
def generate_coords_n_save(save_dir, save_name, J, N_iter=10000, init_sequence=None,beta=1,nb_PCA_comp=0,PCA_comp_list=np.array([]),J_PCA=None,beta_PCA=1):
    """
    Generates a set of sequences using the PLM and saves them both as a numpy file and a text file containing the corresponding letter sequences.
    Saves:
    - A `.npy` file containing the generated sequences in numerical format.
    - A `.txt` file containing the generated sequences in letter format.
    """
    PCA_coords = generate_PCA_coords(init_sequence=init_sequence, N_iter=N_iter, J_PCA=J_PCA,
                                     nb_PCA_comp=nb_PCA_comp,
                                     beta_PCA=beta_PCA,
                                     J=J,
                                     PCA_comp_list=PCA_comp_list, beta=beta)

    PCA_coords = np.array(PCA_coords)

    if nb_PCA_comp != 0:
        save_dir = save_dir + "_PCA_coord"
        save_name = save_name + "_PCA_coord"
        for i in PCA_coords[0]:
            save_dir += str(i) + "_"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save as .npy
    np.save(f"{save_dir}/{save_name}.npy", PCA_coords)

    # Save as .txt
    with open(f"{save_dir}/{save_name}.txt", "w") as f:
        for coords in PCA_coords:
            line = ' '.join(map(str, coords))
            f.write(f"{line}\n")

    print(f"PCA coordinates saved to {save_dir}")


def generate_multiple_targets_n_save(save_dir, save_name_prefix, J, target_coords_list, N_seqs=1000, init_sequence=None, beta=1, nb_PCA_comp=0, J_PCA=None, beta_PCA=1):
    """
    For each target coordinate in the target_coords_list, generate N_seqs sequences using PLM and save them.

    Args:
        save_dir (str): Directory to save outputs.
        save_name_prefix (str): Prefix for filenames.
        J, beta, etc.: Model parameters.
        target_coords_list (list): List of target coordinates, each to be used for sequence generation.
    """
    for idx, target_coords in enumerate(target_coords_list):
        # Generate sequences
        gen_sequences = generate_plm(J, N_seqs, init_sequence, beta=beta, nb_PCA_comp=nb_PCA_comp, PCA_comp_list=target_coords, J_PCA=J_PCA, beta_PCA=beta_PCA)
        
        gen_sequences_letters = [nums_to_letters(sequence, nb_PCA_comp) for sequence in gen_sequences]
        print(f"[{idx}] Generated sequences for target {target_coords} (first 5): {gen_sequences_letters[:5]}")

        # Save directory and file name adjustments
        subdir = f"{save_dir}/target_" + "_".join(map(str, target_coords))
        filename = f"{save_name_prefix}_target_" + "_".join(map(str, target_coords))

        if nb_PCA_comp != 0:
            subdir += "_PCAcomp"
            filename += "_PCAcomp"
            for i in gen_sequences[0, len(gen_sequences[0]) - nb_PCA_comp:]:
                subdir += str(i) + "_"

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        # Save in .npy format
        np.save(f"{subdir}/{filename}.npy", gen_sequences)

        # Save in .txt format
        with open(f"{subdir}/{filename}.txt", "w") as f:
            for sequence in gen_sequences_letters:
                f.write(f"{sequence}\n")

