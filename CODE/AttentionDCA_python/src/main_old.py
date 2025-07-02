from attention import trainer
#from attention_interaction import trainer_interaction
from model import AttentionModel
from utils import load_matrix_3d_from_julia
import os
import torch
import random
import numpy as np

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()    

print(torch.cuda.is_available())
cwd = os.getcwd()

H = 80
d = 23
n_epochs = 500

family = 'HK-RR'
filename = cwd + '/CODE/DataAttentionDCA/HK-RR/HK-RR_174_train.fasta'

model = trainer(n_epochs=n_epochs, H= H, d=d , batch_size=1000, filename= filename, structfile= None, losstype='without_J')
#model = trainer_interaction(n_epochs=n_epochs, L_A=63, H= H, d=d , batch_size=1000, filename= filename, structfile= None)
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
simul_name = '{H}_{d}_{family}_{losstype}_{epochs}'.format(H=H, d=d, family=family,  losstype='without_J', epochs=n_epochs)

# create the result repository and a repository for the model with inside each tensor Q, K, V files
os.makedirs('./results/'+simul_name, exist_ok=True)
save_tensor_to_txt(model.Q.data, "./results/"+simul_name+"/Q_tensor.txt")
save_tensor_to_txt(model.K.data, "./results/"+simul_name+"/K_tensor.txt")
save_tensor_to_txt(model.V.data, "./results/"+simul_name+"/V_tensor.txt")


