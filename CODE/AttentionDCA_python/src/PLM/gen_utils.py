import numpy as np
import platform
import torch
import os
import sys
import matplotlib.pyplot as plt
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from model import AttentionModel
from model_PCA_correlation import AttentionModel_PCA
from dcascore import *
from utils import read_fasta_alignment, remove_duplicate_sequences, add_PCA_coords, add_coords_flat

# back to original path (in PLM)
sys.path.pop(0)  # Removes the parent_dir from sys.path
from model import AttentionModel

from plm_gen_methods import generate_plm_n_save, generate_coords_n_save, generate_multiple_targets_n_save
from seq_utils import read_tensor_from_txt, set_seed, letters_to_nums, modify_seq, find_target_seq, find_all_seqs_with_coord,flatten, load_train_seqs, load_gen_seqs
from PCA_func import plot_projected_pca_time, sequences_from_fasta
from plm_model import SequencePLM

#--------------- FUNCTIONS TO LOAD TENSORS ----------------
def load_J_tensor(model=1):
    """
    Load Q, K, V matrices from jdoms (after training) and compute Jtens.
    """
    set_seed()
    H = 64
    d= 10
    N = 174
    n_epochs = 500
    nb_PCA_comp=2
    loss_type = 'without_J'
    family = 'jdoms' #'jdoms_bacteria_train2'
    cwd = parent_dir
    if model == 1:
        Q_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_brute_force_35_bins/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        K_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_brute_force_35_bins/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        V_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_brute_force_35_bins/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    elif model == 2 or model == 3:
        Q_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_youss/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        K_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_youss/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        V_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_youss/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    H,d,N=Q_1.shape()
    q=V_1.shape[1]
    model=AttentionModel(H,d,N,q,Q=Q_1,V=V_1,K=K_1)
    torch.sum(model.Q-Q_1)
    device = Q_1.device
    L = Q_1.shape[-1]
    W=attention_heads_from_model(model,Q_1,K_1,V_1)
    print(W.shape)

    i_indices = torch.arange(L, device=device).unsqueeze(1)
    j_indices = torch.arange(L, device=device).unsqueeze(0)
    mask = (i_indices != j_indices).float().unsqueeze(0)  # shape (1, L, L)
    W = W * mask

    # Compute Jtens
    Jtens = torch.einsum('hri,hab->abri', W, V_1)  # Shape: (q, q, L, L)
    q = Jtens.shape[0]
    N = Jtens.shape[2]
    print(q)
    print(N)
    print(Jtens.shape)
    return Jtens

def load_JPCA_tensor(model=2):

    set_seed()
    H = 64
    d= 10
    N = 174
    n_epochs = 500
    nb_PCA_comp=2
    loss_type = 'without_J'
    family = 'jdoms' #'jdoms_bacteria_train2'
    cwd = parent_dir
    if model == 2:
        Q_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_2models_35_bins/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        K_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_2models_35_bins/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        V_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_2models_35_bins/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    elif model == 3:
        Q_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_2models_flat_35_bins/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        K_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_2models_flat_35_bins/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
        V_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_PCA_2models_flat_35_bins/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    else:
        raise ValueError("Invalid model type. Choose 2 or 3.")
    H,d,N1=Q_1.shape
    _,_,N2=K_1.shape
    _,q1,q2=V_1.shape
    model=AttentionModel_PCA(H,d,N1,N2,q1,q2,Q=Q_1,V=V_1,K=K_1)
    torch.sum(model.Q-Q_1)
    device = Q_1.device
    L = Q_1.shape[-1]
    W=attention_heads_from_model(model,Q_1,K_1,V_1)
    print(W.shape)

    # Compute Jtens
    Jtens_PCA = torch.einsum('hri,hab->abri', W, V_1)  # Shape: (q, q, L, L)
    print(Jtens_PCA.shape)
    return Jtens_PCA




#--------------- FUNCTIONS TO GENERATE SEQS ----------------
def generate_random(betas, betas_PCA, N_seqs, target_coords_list, model=2, save_dir=None):
    """
    Generate random sequences with specified betas and save them.
    """
    if save_dir is None:
        save_dir = f"generated_sequences_Model_{model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Jtens = load_J_tensor(model=model)
    Jtens_PCA = None
    nb_PCA_comps = 2
    if model != 1:
        Jtens_PCA = load_JPCA_tensor(model=model)
    if model ==3:
        target_coords_list = [flatten(coords, nb_bins_PCA=35) for coords in target_coords_list]
        print("target coords flat list:", target_coords_list)
        nb_PCA_comps = 1
    for target_coords in target_coords_list:
        target_coords = np.atleast_1d(target_coords)
        target_coords_str = '_'.join(str(int(x)) for x in target_coords)
        print(f"Generating sequences with target coordinates: {target_coords}")
        for beta in betas:
            for beta_PCA in betas_PCA:
                save_name = f"gen_seqs_randinit_Ns{N_seqs}_b{beta}_b_PCA{beta_PCA}_{target_coords_str}"
                print(f"Generating sequences with beta={beta}, beta_PCA={beta_PCA}")
                print(target_coords)
                print("shape of target_coords:", target_coords.shape)
                generate_plm_n_save(save_dir, save_name, Jtens, N_seqs, init_sequence=None, beta=beta, nb_PCA_comp=nb_PCA_comps, J_PCA=Jtens_PCA, beta_PCA=beta_PCA, PCA_comp_list=target_coords)


#--------------- FUNCTIONS TO PLOT PCA SEQUENCES ----------------

def plot_pca_results(file_dir,file_name, target_coords = None, Nbins=35, mac=True):
    train_sequences = load_train_seqs(mac=mac)
    if file_name is not None:
        gen_sequences = load_gen_seqs(file_dir, file_name, mac=mac)
        plot_projected_pca_time(sequences_reference=train_sequences, sequences_to_project=gen_sequences, title="PCA Projection: Train vs Generated", Nbins=Nbins, target_coords=target_coords)
        return 0
    else: # loop all files in the directory
        for file_name in os.listdir(file_dir):
            if file_name.endswith('.npy'):
                gen_sequences, target_coords = load_gen_seqs(file_dir, file_name, mac=mac)
                print(f"Processing file: {file_name}")
                plot_projected_pca_time(sequences_reference=train_sequences, sequences_to_project=gen_sequences, title=f"PCA Projection: Train vs {file_name}", Nbins=Nbins, target_coords=target_coords)
        return 0


#--------------- FUNCTIONS FOR PCA COORDS ----------------
def load_target_seq(target):
    if platform.system() == "Darwin":  # macOS
        base_path = '/Users/marzioformica/Desktop/EPFL/Master/StageLBS/PCA_gen_AI/CODE/DataAttentionDCA/jdoms'
    else:  # assume Windows for Youss
        base_path = r"C:\Users\youss\OneDrive\Bureau\master epfl\MA2\TP4 De los Rios\git_test\PLM-gen-DCA\Attention-DCA-main\CODE\DataAttentionDCA\jdoms"

    file_test_data = f"{base_path}/jdoms_bacteria_train2.fasta"
    seq_data_test=read_fasta_alignment(file_test_data,0.8)

    print(seq_data_test.shape)
    seq_data_test_filtered,_=remove_duplicate_sequences(seq_data_test)
    seq_data_test_filtered=seq_data_test_filtered.T
    seq_data_test_filtered_with_PCA= add_PCA_coords(seq_data_test_filtered,35) #####DONT FORGET TO CHANGE NUMBER OF BINS IF NECESSARY
    init_sequence_num = find_target_seq(target, seq_data_test_filtered_with_PCA)
    return init_sequence_num

def load_target_seqs(target_coord, flat=False):
    if platform.system() == "Darwin":  # macOS
        base_path = '/Users/marzioformica/Desktop/EPFL/Master/StageLBS/PCA_gen_AI/CODE/DataAttentionDCA/jdoms'
    else:  # Windows
        base_path = r"C:\Users\youss\OneDrive\Bureau\master epfl\MA2\TP4 De los Rios\git_test\PLM-gen-DCA\Attention-DCA-main\CODE\DataAttentionDCA\jdoms"

    file_test_data = f"{base_path}/jdoms_bacteria_train2.fasta"
    seq_data_test = read_fasta_alignment(file_test_data, 0.8)

    print(seq_data_test.shape)
    seq_data_test_filtered, _ = remove_duplicate_sequences(seq_data_test)
    seq_data_test_filtered = seq_data_test_filtered.T
    if flat:
        seq_data_test_filtered_with_PCA = add_coords_flat(seq_data_test_filtered, 35)
    else:
        seq_data_test_filtered_with_PCA = add_PCA_coords(seq_data_test_filtered, 35)
    return find_all_seqs_with_coord(target_coord, seq_data_test_filtered_with_PCA)



def plot_energy_from_multiple_seqs(target, model=1, b=1, b_PCA=1, Nbins=35):
    Jtens = load_J_tensor(model=model)
    Jtens_PCA = None
    target_x, target_y = target
    nb_PCA_comp = 2
    if model != 1:
        Jtens_PCA = load_JPCA_tensor(model=model)
    if model == 3:
        nb_PCA_comp = 1
        target_flat = flatten(target, nb_bins_PCA=Nbins)
        print("target flat:", target_flat)
        init_sequences_num = load_target_seqs(target_flat, flat=True)
    else:
        init_sequences_num = load_target_seqs(target)
    energies_2D = np.zeros((Nbins, Nbins))  # Initialize energy landscape
    for init_sequence_num in init_sequences_num:
        #target = init_sequence_num[-nb_PCA_comp:]  # last two values are PCA coordinates
        #target_x, target_y = init_sequence_num[-nb_PCA_comp:]
        seq = SequencePLM(J=Jtens, initial_sequence=init_sequence_num, beta=b, beta_PCA=b_PCA, nb_PCA_comp=nb_PCA_comp, J_tens_PCA=Jtens_PCA)
        energies_2D += seq.compute_coord_energy(model=model)
    energies_2D /= len(init_sequences_num) 

    plt.figure(figsize=(8, 6))
    #cmap: opposite of viridis
    plt.imshow(energies_2D.T, origin='lower', cmap='viridis_r', extent=(0, Nbins, 0, Nbins), aspect='auto')
    plt.colorbar(label='Energy')
    plt.scatter(target_x, target_y, c='red', s=100, edgecolors='black', label=f'True coord: ({target_x}, {target_y})')
    plt.title("Energy Landscape")
    plt.xlabel("PCA bin x")
    plt.ylabel("PCA bin y")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


#def plot_energy_from_multiple_seqs(target, model=1, b=1, b_PCA=1, Nbins=35):
#    print(f"function called with model {model}")
#    if model == 3:
#        print("Model 3")
#    else:
#        print("Not model 3")
#    Jtens = load_J_tensor(model=model)
#    Jtens_PCA = None
#    target_x, target_y = target
#    nb_PCA_comp = 2
#    print("Model:", model)
#    if model != 1:
#        Jtens_PCA = load_JPCA_tensor(model=model)
#    init_sequences_num = load_target_seqs(target)
#    if model == 3:
#        print("Model 3")
#    else:
#        print("Not model 3")
#    if model == 3:
#        print("Model 3")
#        nb_PCA_comp = 1
#        target_flat = flatten(target, nb_bins_PCA=Nbins)
#        print("target flat:", target_flat)
#        init_sequences_num = 0
#        init_sequences_num = load_target_seqs(target_flat)
#
#    energies_2D = np.zeros((Nbins, Nbins))  # Initialize energy landscape
#    for init_sequence_num in init_sequences_num:
#        #target = init_sequence_num[-nb_PCA_comp:]  # last two values are PCA coordinates
#        #target_x, target_y = init_sequence_num[-nb_PCA_comp:]
#        seq = SequencePLM(J=Jtens, initial_sequence=init_sequence_num, beta=b, beta_PCA=b_PCA, nb_PCA_comp=nb_PCA_comp, J_tens_PCA=Jtens_PCA)
#        energies_2D += seq.compute_coord_energy(model=model)
#    energies_2D /= len(init_sequences_num)  # Average over all sequences
#
#    plt.figure(figsize=(8, 6))
#    #cmap: opposite of viridis
#    plt.imshow(energies_2D.T, origin='lower', cmap='viridis_r', extent=(0, Nbins, 0, Nbins), aspect='auto')
#    plt.colorbar(label='Energy')
#    plt.scatter(target_x, target_y, c='red', s=100, edgecolors='black', label=f'True coord: ({target_x}, {target_y})')
#    plt.title("Energy Landscape")
#    plt.xlabel("PCA bin x")
#    plt.ylabel("PCA bin y")
#    plt.legend()
#    plt.grid(False)
#    plt.tight_layout()
#    plt.show()


def generate_coords_func(target_coords, N_iter, beta=1, beta_PCA=1, model=2, save_dir=None, Nbins=35):
    """
    Generate random sequences with specified betas and save them.
    """
    if save_dir is None:
        save_dir = f"generated_coords_PCA_Model_{model}"
    set_seed()
    Jtens = load_J_tensor(model=model)
    Jtens_PCA = None
    nb_PCA_comp = 2
    if model != 1:
        Jtens_PCA = load_JPCA_tensor(model=model)
    if model ==3:
        nb_PCA_comps = 1
    init_seq_num = load_target_seq(target_coords)
    target = init_seq_num[-nb_PCA_comps:]
    save_name = f"gen_coords_Ns{N_iter}_b{beta}_b_PCA{beta_PCA}_{target_coords}"
    print(f"Generating sequences with beta={beta}, beta_PCA={beta_PCA}")
    generate_coords_n_save(save_dir=save_dir, save_name=save_name, J=Jtens, N_iter=N_iter, init_sequence=init_seq_num, nb_PCA_comp=nb_PCA_comp, J_PCA=Jtens_PCA, beta_PCA=beta_PCA, model=model)
    # PLOT
    cwd = os.getcwd()
    coords_flat = np.load(cwd+f"/{save_dir}_PCA_coord/{save_name}_PCA_coord.npy")  # shape: (N_iter,2)
    print("coords_flat shape:", coords_flat.shape)
    # Unflatten
    x_coords = coords_flat[:, 0]  # x-coordinates
    y_coords = coords_flat[:, 1]  # y-coordinates
    # 2D colormap grid
    heatmap = np.zeros((Nbins, Nbins), dtype=int)
    # Populate the heatmap with visit counts
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < Nbins and 0 <= y < Nbins:  # Ensure coordinates are within bounds
            heatmap[y, x] += 1  # Increment visit count for the bin

    # plotting
    # --- Target coordinates (from init_sequence_num)
    target_x, target_y = target_coords.flatten() #init_sequence_num[-2:]
    print("target coordinates:", target_x, target_y)

    # --- Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, origin='lower', cmap='viridis')
    plt.colorbar(label='Number of visits')
    plt.scatter(target_x, target_y, c='red', s=100, edgecolors='black', label=f'True coord: ({target_x}, {target_y})')
    plt.title(f"Model {model}")
    plt.xlabel("PCA bin x")
    plt.ylabel("PCA bin y")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()