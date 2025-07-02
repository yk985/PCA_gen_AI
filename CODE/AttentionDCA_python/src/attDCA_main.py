from attention import trainer
from model import AttentionModel
from utils import load_matrix_3d_from_julia
import os
import torch
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("H", type=int, help="Number of attention heads")
parser.add_argument("d", type=int, help="Internal dimension")
parser.add_argument("l", type=float, help="Regularization parameter")
parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs")
args = parser.parse_args()

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()    

cwd = os.getcwd()

H = args.H
d = args.d
l=args.l
n_epochs = args.n_epochs

family = 'jdoms_bacteria'
filename = cwd + '/CODE/DataAttentionDCA/jdoms/jdoms_bacteria_train2.fasta'

print(f"Training model with H={H}, d={d}, lambda={l}, n_epochs={n_epochs} on family {family} using file {filename}")

model = trainer(n_epochs=n_epochs, H= H, d=d , batch_size=1000, lambd=l, filename= filename, structfile= None, losstype='without_J')

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


#specify the simultation name with epochs, H, d, filename, losstype
simul_name = '{H}_{d}_{family}_{l}_{epochs}'.format(H=H, d=d, family=family,  l=l, epochs=n_epochs)

# create the result repository and a repository for the model with inside each tensor Q, K, V files
os.makedirs('./results/'+simul_name, exist_ok=True)
save_tensor_to_txt(model.Q.data, "./results/"+simul_name+"/Q_tensor.txt")
save_tensor_to_txt(model.K.data, "./results/"+simul_name+"/K_tensor.txt")
save_tensor_to_txt(model.V.data, "./results/"+simul_name+"/V_tensor.txt")
