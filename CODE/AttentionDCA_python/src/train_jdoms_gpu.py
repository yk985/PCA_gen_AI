from attention import trainer, trainer_PCA_comp_brute_force, trainer_PCA_comp_2_model
from attetion_sep_size_heads import trainer_multidomain_strategyB

import os
import torch
import random
import numpy as np
import sys
import h5py
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..','src'))
sys.path.insert(0, parent_dir)
from model import AttentionModel
from model_PCA_correlation import AttentionModel_PCA
from dcascore import *
from utils import read_fasta_alignment, remove_duplicate_sequences, add_PCA_coords

# back to original path (in PLM)
sys.path.pop(0)  # Removes the parent_dir from sys.path
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



cwd = os.getcwd()
def save_tensor_to_txt(tensor, filename):
    with open(filename, 'w') as f:
        # Write tensor dimensions
        dims = tensor.size()
        f.write(" ".join(map(str, dims)) + "\n")

        # Iterate over the first dimension (slices)
        for i in range(dims[0]):
            f.write("\n")
            f.write(f"Slice {i + 1}\n")
            for j in range(dims[1]):  # Iterate over the second dimension (rows)
                row = tensor[i, j].tolist()
                f.write(",".join(map(str, row)) + "\n")
def main1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    family = 'jdoms'
    H = 64
    d= 10
    N = 174
    n_epochs = 500
    nb_PCA_comp=50
    loss_type = 'without_J'
    family = 'jdoms' #'jdoms_bacteria_train2'
    cwd = parent_dir
    Q_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_mod_alessandro/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    K_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_mod_alessandro/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    V_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_mod_alessandro/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
    Q_1=Q_1.to(device)
    K_1=K_1.to(device)
    V_1=V_1.to(device)
    cwd=cwd.replace(r"\CODE\AttentionDCA_python\src",'')

    nb_bins_PCA=17
    H = 64
    d= 10
    n_epochs = 500 #could go more down
    #domain1_end = 63 #if your protein msa input family has a domain division, this is the zero index of the last aminoacid of the first domain
    domain1_end = 62
    filename = cwd + r'\CODE\DataAttentionDCA\jdoms\jdoms_bacteria_train2.fasta' #new lisa for energy couplings: HKRR174
    structfile = None
    trainer_type = 'std' # 'std' or 'std_with_masks' 'multidomain'


    if trainer_type == 'std':
        domain1_end = 0
        H1 = H2 = 0


    if trainer_type == 'std' or trainer_type == 'std_with_masks':
        
        model = trainer_PCA_comp_2_model(
            n_epochs,
            Q_1,
            K_1,
            V_1,
            H=H,
            d=d,
            filename=filename,
            structfile=structfile,
            losstype='without_J',
            index_last_domain1=domain1_end,  # this value is the 0-index include the domain 1, for HK-RR is 63 (so 64 long domain 1) 
            #it is set to zero if i dont want to divide any domain
            H1 = H1,
            H2 = H2,
            max_gap_frac=0.8,
            batch_size=250,
            n_comp_pca=nb_PCA_comp,
            nb_bins_PCA=nb_bins_PCA
        )

        # Create results directory
        simul_name = f'{H}_{d}_{family}_without_J_{n_epochs}_2model_pretrained_n_pca_{nb_PCA_comp}'
        results_dir = f'./results/{simul_name}'
        os.makedirs(results_dir, exist_ok=True)

        # Save model parameters
        save_tensor_to_txt(model.Q.data, "./results/"+simul_name+"/Q_tensor.txt")
        save_tensor_to_txt(model.K.data, "./results/"+simul_name+"/K_tensor.txt")
        save_tensor_to_txt(model.V.data, "./results/"+simul_name+"/V_tensor.txt")

if __name__ == "__main__":
    # Needed for Windows
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")  # optional but safe
    main1()