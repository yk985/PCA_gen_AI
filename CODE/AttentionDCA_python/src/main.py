from attention import trainer
from attetion_sep_size_heads import trainer_multidomain_strategyB

import os
import torch
import random
import numpy as np

cwd = os.getcwd()

##############################################################HELPER FUNCTIONS: RANDOM INIZIALIZATIONS AND SAVING/LOADING TENSORS
from utils import load_matrix_3d_from_julia #if needed to load initial model from julia generated matrices
def read_tensor_from_txt(filename):
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

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed=0)    

#uncomment if you have an initial model
# # Load the matrices to check coerence with julia code (now not available anymore in this repository)
# Q_np = load_matrix_3d_from_julia(cwd + "/tensor_initial_models/new_Q_values_32H_23d.txt")
# K_np = load_matrix_3d_from_julia(cwd + "/tensor_initial_models/new_K_values_32H_23d.txt")
# V_np = load_matrix_3d_from_julia(cwd + "/tensor_initial_models/new_V_values_32H_22q.txt")  # Updated to load as 3D

# H = len(Q_np)
# d = len(Q_np[0])
# N = len(Q_np[0][0])
# q = len(V_np[0][0])

# # Convert NumPy arrays to PyTorch tensors
# Q_tensor = torch.from_numpy(Q_np).float()  # Shape: (32, 23, 53)
# K_tensor = torch.from_numpy(K_np).float()  # Shape: (32, 23, 53)
# V_tensor = torch.from_numpy(V_np).float()  # Shape: (23, 22, 22)



# init_m = AttentionModel(H, d, N, q)
# init_m.Q.data = Q_tensor
# init_m.K.data = K_tensor
# init_m.V.data = V_tensor

# Optional: Verify tensor shapes and data types
#print(f"Shape of Q_tensor: {Q_tensor.shape}")  # Expected: torch.Size([2, 3, 53])
#print(f"Shape of K_tensor: {K_tensor.shape}")  # Expected: torch.Size([2, 3, 53])
#print(f"Shape of V_tensor: {V_tensor.shape}")  # Expected: torch.Size([2, 22, 22])


#################################################ARCHITECTURE AND TRAINING SELECTION

H = 64
d= 10
n_epochs = 500 #could go more down
#domain1_end = 63 #if your protein msa input family has a domain division, this is the zero index of the last aminoacid of the first domain
domain1_end = 62
#family = 'HK-RR_w_mask64_11_14_9_lisa_energy_coupling'
#family = 'provo_new_cleaned_model_conPF'
#family = 'HKRR_30_10_80_JUSTdomain1_withHKRRtrainingfasta_d23_500batch'
#family = 'HKRR_25_15_160_JUSTdomain1to2update_withHKRRtrainingfasta_d10_500batch_REDO'
family = 'HKRR_masked_22_26_16'

#filename = cwd + '/CODE/DataAttentionDCA/data/PF00076/PF00076_mgap6.fasta.gz'
#filename = cwd + '/CODE/DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz'
#structfile = cwd + '/CODE/DataAttentionDCA/data/PF00014/PF00014_struct.dat'
#filename = cwd + '/CODE/DataAttentionDCA/data/lisa_data/HK_RR_filtered_sequences2.fasta' #this is some sort of think I made
#filename = cwd + '/CODE/DataAttentionDCA/data/lisa_data/HK-RR_concatenated_nodupli.fasta' #USE THIS FOR HKRR176
#filename = cwd + '/CODE/DataAttentionDCA/data/lisa_data/HK_in_Concat_nnn.fasta'
filename = cwd + '/CODE/DataAttentionDCA/data/lisa_data/HK-RR_174_train.fasta' #new lisa for energy couplings: HKRR174
structfile = None
# H1 = 11
# H2 = 15+11
#H1= 25
#H2 =15+H1
H1 = 22
H2 = 26 + H1
trainer_type = 'std_with_masks' # 'std' or 'std_with_masks' 'multidomain'

if trainer_type == 'std':
    domain1_end = 0
    H1 = H2 = 0

if trainer_type == 'multidomain':
    assert(np.mod(H-H2,2)==0) #this is to check that the number of heads is even, so that the division intersection heads and its reciprocal is not possible


#standard training/model with or without masks
if trainer_type == 'std' or trainer_type == 'std_with_masks':
    
    model = trainer(
        n_epochs=n_epochs,
        H=H,
        d=d,
        filename=filename,
        structfile=structfile,
        losstype='without_J',
        index_last_domain1=domain1_end,  # this value is the 0-index include the domain 1, for HK-RR is 63 (so 64 long domain 1) 
        #it is set to zero if i dont want to divide any domain
        H1 = H1,
        H2 = H2
    )

    # Create results directory
    simul_name = f'{H}_{d}_{family}_without_J_{n_epochs}'
    results_dir = f'./results/{simul_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Save model parameters
    save_tensor_to_txt(model.Q.data, "./results/"+simul_name+"/Q_tensor.txt")
    save_tensor_to_txt(model.K.data, "./results/"+simul_name+"/K_tensor.txt")
    save_tensor_to_txt(model.V.data, "./results/"+simul_name+"/V_tensor.txt")

#multidomain training/model: each domain has its own attention block (with size specific heads) with shared interdomain attention block
if trainer_type == 'multidomain':
    other_info_mat_ene = False # give pretrained model to interdomain subblock
    model = trainer_multidomain_strategyB(n_epochs=n_epochs,
                                     H=H, d=d,
                                     batch_size=500,
                                     eta=0.005, lambd=0.001,
                                     domain1_end=domain1_end,
                                     H1 = H1,
                                     H2=H2,
                                     filename=filename,
                                     other_info_mat_ene = other_info_mat_ene)
    # Create results directory
    simul_name = f'{H}_{d}_{family}_without_J_{n_epochs}'
    results_dir = f'./results/{simul_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Save model parameters
    save_tensor_to_txt(model.Q2.data, "./results/"+simul_name+"/Q2_tensor.txt")
    save_tensor_to_txt(model.K2.data, "./results/"+simul_name+"/K2_tensor.txt")
    save_tensor_to_txt(model.V2.data, "./results/"+simul_name+"/V2_tensor.txt")

    save_tensor_to_txt(model.Q1.data, "./results/"+simul_name+"/Q1_tensor.txt")
    save_tensor_to_txt(model.K1.data, "./results/"+simul_name+"/K1_tensor.txt")
    save_tensor_to_txt(model.V1.data, "./results/"+simul_name+"/V1_tensor.txt")

    save_tensor_to_txt(model.Qint1.data, "./results/"+simul_name+"/Qint1_tensor.txt")
    save_tensor_to_txt(model.Kint1.data, "./results/"+simul_name+"/Kint1_tensor.txt")
    save_tensor_to_txt(model.Vint1.data, "./results/"+simul_name+"/Vint1_tensor.txt")

    save_tensor_to_txt(model.Qint2.data, "./results/"+simul_name+"/Qint2_tensor.txt")
    save_tensor_to_txt(model.Kint2.data, "./results/"+simul_name+"/Kint2_tensor.txt")
    save_tensor_to_txt(model.Vint2.data, "./results/"+simul_name+"/Vint2_tensor.txt")