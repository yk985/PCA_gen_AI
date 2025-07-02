import numpy as np
import torch
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from model import AttentionModel
from dcascore import *
# back to original path (in PLM)
sys.path.pop(0)  # Removes the parent_dir from sys.path
from model import AttentionModel

from plm_gen_methods import generate_plm_n_save
from seq_utils import read_tensor_from_txt, set_seed, letters_to_nums, modify_seq


#---------------------- Choose the method to generate sequences, comment rest ----------------------


##############################################################
"""
    Load Q, K, V matrices from jdoms (after training)
"""
set_seed()
H = 64
d= 10
N = 174
n_epochs = 500
loss_type = 'without_J'
family = 'jdoms' #'jdoms_bacteria_train2'
cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main/CODE/AttentionDCA_python/src'
Q_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_youss/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
K_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_youss/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
V_1 = read_tensor_from_txt( cwd +"/results/{H}_{d}_{family}_{losstype}_{n_epochs}_youss/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
H,d,N=Q_1.shape
q=V_1.shape[1]

##############################################################
"""
    Initialize the model and compute couplings J from Q, K, V
""" 
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

##############################################################
"""
    Generate sequences with PLM random initialization
"""
save_dir = "generated_sequences"
N_seqs = 40000
save_name = "generated_sequences_randinit_40000"
#generate_plm_n_save(save_dir, save_name, Jtens, N_seqs=40000, init_sequence=None)

##############################################################
"""
    Generate sequences with PLM initialization from a sequence
"""
init_sequence = 'DYYQVLGVPKDADAKSIKKAFRKLARKYHPDVNPGDKEAERKFKEANEANEVLSDPEKRKKYD'
init_sequence_num = letters_to_nums(init_sequence)
ratio = 0.1
init_sequence_num = modify_seq(init_sequence_num, ratio)
N_seqs=40000
save_name = f"gen_seqs_w_init_seq_Ns{N_seqs}_r{ratio}"

generate_plm_n_save(save_dir, save_name, Jtens, N_seqs, init_sequence=init_sequence_num)

##############################################################
"""
    Generate sequences alternate method with PLM initialization from a sequence
"""

N_seqs=10000
N_iters=2000
save_name = f"gen_seqs_alter_randinit_Ns{N_seqs}_Ni{N_iters}"
#generate_plm_alter_n_save(save_dir, save_name, Jtens, N_seqs, N_iters, init_sequence=None)

##############################################################
"""
    Generate sequences with PLM initialization from a sequence different betas
"""
init_sequence = 'DYYQVLGVPKDADAKSIKKAFRKLARKYHPDVNPGDKEAERKFKEANEANEVLSDPEKRKKYD'
init_sequence_num = letters_to_nums(init_sequence)
ratio = 0.1
init_sequence_num = modify_seq(init_sequence_num, ratio)

N_seqs=4000
betas = [0.01, 0.1, 0.5, 1, 2, 4, 10]
for b in betas:
    save_name = f"gen_seqs_w_init_seq_Ns{N_seqs}_r{ratio}_b{b}"

    generate_plm_n_save(save_dir, save_name, Jtens, N_seqs, init_sequence=init_sequence_num, beta=b)
    #generate_plm_n_save(save_dir, save_name, Jtens, N_seqs, init_sequence=init_sequence_num, beta=b)