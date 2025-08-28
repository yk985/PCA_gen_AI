import site
from tqdm import tqdm
import numpy as np
import os
from seq_utils import letter_to_num
from tqdm import tqdm
from scipy.special import softmax
class SequenceAR:
    def __init__(self,J,N_seq,p_0,beta=1):
        self.J=J
        self.N_seq=N_seq
        self.L=J.shape[-1]
        self.beta=beta
        self.q=J.shape[0]
        self.seq_array=np.zeros((N_seq,self.L))
        self.p0=p_0

    def energy_position_k(self,k, aa):
        n_seq = self.seq_array.shape[0]
        # indices of positions < k
        idx = np.arange(k)

        # shape (n_seq, k)
        seq_sub = self.seq_array[:, idx]
        print(seq_sub.shape)

        # gather J[k, j, aa, seq_sub] for all j<k, seqs
        # this broadcasts: (k,) vs (n_seq, k)
        e = self.J[aa, seq_sub,k, idx[None, :]]
        print(e.shape)

        # sum over positions <k, leaving (n_seq,)
        return e.sum(axis=1)
    
    def ene_J_autre(self,J_autre,k, aa):

        idx = np.arange(k)

        # shape (n_seq, k)
        seq_sub = self.seq_array[:, idx]
        

        # gather J[k, j, aa, seq_sub] for all j<k, seqs
        # this broadcasts: (k,) vs (n_seq, k)
        e2 = J_autre[aa, seq_sub,k, idx[None, :]]
        e2=e2.sum(axis=1)
        

        # sum over positions <k, leaving (n_seq,)
        return  e2


    def prob_per_position(self,site):
        all_proba=np.zeros((self.N_seq,self.q))
        for aa in range(self.q):
            all_proba[:,aa]=self.beta*self.energy_position_k(site,aa)
        all_proba=softmax(all_proba,axis=1)
        return all_proba
    
    def compare_J(self,J_autre,site):
        all_p_old=self.prob_per_position(site)
        all_proba=np.zeros((self.N_seq,self.q))
        for aa in range(self.q):
            all_proba[:,aa]=self.beta*self.ene_J_autre(J_autre,site,aa)
        all_proba=softmax(all_proba,axis=1)
        diff_proba=abs(all_p_old-all_proba).mean().item()
        return diff_proba
    
    def gen_seq(self):
        self.seq_array[:,0]=np.random.choice(np.arange(self.q),self.N_seq,replace=True,p=self.p0)
        for site in tqdm(range(1,self.L)):
            all_proba_site=self.prob_per_position(site)
            u = np.random.rand(self.N_seq, 1)  # (N_seq, 1) uniform samples
            cum_probs = np.cumsum(all_proba_site, axis=1)  # (N_seq, q)
            choices = (u < cum_probs).argmax(axis=1)
            self.seq_array[:, site] = choices

    def save_sequence(self, save_dir, filename):
        """Save sequence array as a compressed .npz file."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename + ".npz")
        np.savez_compressed(path, seq_array=self.seq_array)

    def load_sequence(self, save_dir, filename):
        """Load sequence array from a compressed .npz file."""
        path = os.path.join(save_dir, filename + ".npz")
        data = np.load(path)
        self.seq_array = data["seq_array"]
        return self.seq_array
    
   


        
        
        