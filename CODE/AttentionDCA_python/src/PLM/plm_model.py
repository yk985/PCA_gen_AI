import site
from tqdm import tqdm
import numpy as np

from seq_utils import letter_to_num

class SequencePLMvec:
    def __init__(self, J, initial_sequence = None, beta = 1, nb_PCA_comp=0,target_coords=np.array([]),J_tens_PCA=None,beta_PCA=1):
        """
        Initialize the SequencePLM object with a coupling tensor J of the family and an optional initial sequence.
        """
        self.J = J
        self.J_PCA=J_tens_PCA
        #self.L = J.shape[-1]
        self.beta = beta
        self.beta_PCA=beta_PCA
        self.nb_PCA_comp=nb_PCA_comp
        if not (J_tens_PCA is None):
            if nb_PCA_comp!=J_tens_PCA.shape[-1]:
                print("Mismatch of PCA tensor and nb PCA components indicated")
        if J_tens_PCA is None:
            self.L = J.shape[-1] - nb_PCA_comp  # Length of the sequence without PCA components
        else:
            self.L = J.shape[-1]
        if initial_sequence is None:
            self.sequence = np.random.choice(np.arange(21), self.L) # Sequence of ints (1 to 21) 
            if len(target_coords)==nb_PCA_comp:
                self.sequence = np.concatenate((self.sequence,target_coords))
            else:
                print("number of PCA components doesn't match size of PCA list")
        else:
            self.sequence = initial_sequence

    def to_letter(self):
        """
        Show sequence as letters
        """
        print("Sequence:", self.sequence)
        num_to_letter = {v: k for k, v in letter_to_num.items()}
        letter_seq = ''.join([num_to_letter[i] for i in self.sequence[:len(self.sequence)-self.nb_PCA_comp]])
        print(letter_seq)
        return letter_seq

    def plm_calc(self, site, trial_aa):
        """
        Compute unnormalized pseudo-likelihood of trial_aa at a given site.
        site: int from 0 to L-1
        trial_aa: int from 0 to 21 (amino acid index)
        """
        if site < 0 or site >= self.L:
            raise ValueError(f"Site {site} is out of bounds for sequence length {self.L}.")
        sum_energy = 0.0
        if not ( self.J_PCA is None):      
            for i in range(self.nb_PCA_comp):
                    PCA_coord = self.sequence[self.L + i] 
                    sum_energy += self.beta_PCA * self.J_PCA[trial_aa, PCA_coord, site, i] # J_PCA of shape ([q, Nbins, L, nb_PCA_comp])
        else:
            for i in range(self.nb_PCA_comp):
                sum_energy+=self.beta_PCA*self.J[trial_aa,self.sequence[self.L+i],site,self.L+i] # J of shape ([q, q+Nbins, L, L+nb_PCA_comp])
        for j in range(self.L):
            if j == site:
                continue
            aa_j = self.sequence[j]
            sum_energy += self.beta * self.J[trial_aa, aa_j, site, j] 
        prob = sum_energy  
        return prob
    
    def plm_calc_vec(self, site):
        """
        Compute unnormalized pseudo-likelihoods for ALL trial amino acids at once.
        Returns array of shape (q,) with one score per amino acid.
        """
        q = 21
        trial_aas = np.arange(q)[:, None]  # (21,1)

        # --- Sequence couplings ---
        mask = np.arange(self.L) != site
        other_sites = np.arange(self.L)[mask]
        aa_vector = self.sequence[:self.L][mask]              # amino acids at other sites
        seq_energies = self.J[trial_aas, aa_vector, site, other_sites]  # (21, L-1)
        sum_energy = self.beta * np.sum(seq_energies, axis=1)           # (21,)

        # --- PCA couplings ---
        if self.nb_PCA_comp > 0:
            pca_indices = np.arange(self.L, self.L + self.nb_PCA_comp)  # j-axis indices
            pca_bins = self.sequence[pca_indices]                       # PCA bins

            if self.J_PCA is not None:
                # J_PCA: (q, Nbins, L, nb_PCA_comp)
                pca_energies = self.J_PCA[trial_aas, pca_bins, site, np.arange(self.nb_PCA_comp)]
            else:
                # J: (q, q+Nbins, L, L+nb_PCA_comp) — bins already shifted by q
                pca_energies = self.J[trial_aas, pca_bins, site, pca_indices]

            sum_energy += self.beta_PCA * np.sum(pca_energies, axis=1)

        return sum_energy
    
    def plm_site_distribution(self, site):
        """
        Compute probability distriution for specific site (normalized)
        """
        probs = []
        for trial_aa in range(21):
            probs.append(self.plm_calc(site, trial_aa))
        probs = np.array(probs)
        probs = np.exp(probs-probs.max()) #to avoid overflow for high beta
        probs /= probs.sum()
        return probs
    
    def plm_site_distribution_vec(self, site):
        """
        Compute probability distribution for specific site (normalized).
        """
        sum_energy = self.plm_calc_vec(site)  # (21,)
        probs = np.exp(sum_energy - np.max(sum_energy))  # stable softmax
        probs /= probs.sum()
        return probs
    def check_vectorized(self, site):
        """
        Check that vectorized and loop implementations give same result
        """
        probs_loop = self.plm_site_distribution(site)
        probs_vec = self.plm_site_distribution_vec(site)
        assert np.allclose(probs_loop, probs_vec), f"Mismatch at site {site}"
        print(f"Vectorized check passed at site {site}")

    def draw_aa(self, site, vec=True):
        """
        Sample a new AA at the given site from PLM distribution
        """
        if vec:
            probs = self.plm_site_distribution_vec(site)
        else:
            probs = self.plm_site_distribution(site)
        new_aa = np.random.choice(21, p=probs) # aa from 0 to 20
        self.sequence[site] = new_aa
    
    def seq_energy(self):
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                sum+=self.J[self.sequence[i], self.sequence[j],i,j]
        return sum

class SequencePLM:
    def __init__(self, J, initial_sequence = None, beta = 1, nb_PCA_comp=0,PCA_component_list=np.array([]),J_tens_PCA=None,beta_PCA=1):
        """
        Initialize the SequencePLM object with a coupling tensor J of the family and an optional initial sequence.
        """
        self.J = J
        self.J_PCA=J_tens_PCA
        #self.L = J.shape[-1]
        self.beta = beta
        self.beta_PCA=beta_PCA
        self.nb_PCA_comp=nb_PCA_comp
        if not (J_tens_PCA is None):
            if nb_PCA_comp!=J_tens_PCA.shape[-1]:
                print("Mismatch of PCA tensor and nb PCA components indicated")
        if J_tens_PCA is None:
            self.L = J.shape[-1] - nb_PCA_comp  # Length of the sequence without PCA components
        else:
            self.L = J.shape[-1]
        if initial_sequence is None:
            self.sequence = np.random.choice(np.arange(21), self.L) # Sequence of ints (1 to 21) 
            if len(PCA_component_list)==nb_PCA_comp:
                self.sequence = np.concatenate((self.sequence,PCA_component_list))
            else:
                print("number of PCA components doesn't match size of PCA list")
            #if len(PCA_component_list) == nb_PCA_comp:
            #    if nb_PCA_comp == 1:
            #        print('here')
            #    # Expecting 2D coordinates (x, y), flatten into 1D bin index
            #        first_comp = PCA_component_list[0]
            #        if isinstance(first_comp, (tuple, list, np.ndarray)) and len(first_comp) == 2:
            #            x, y = first_comp
            #            flat_coord = x * self.nb_bins_PCA + y  # flatten 2D coords
            #            self.sequence = np.concatenate((self.sequence, [flat_coord]))
            #        else:
            #        # Already flattened or just one value
            #            self.sequence = np.concatenate((self.sequence, [first_comp]))
            #    else:
            #    # nb_PCA_comp > 1: assume already in flattened or separate components
            #        self.sequence = np.concatenate((self.sequence, PCA_component_list))
            #else:
            #    raise ValueError("Number of PCA components doesn't match PCA_component_list")
        else:
            self.sequence = initial_sequence

    def to_letter(self):
        """
        Show sequence as letters
        """
        print("Sequence:", self.sequence)
        num_to_letter = {v: k for k, v in letter_to_num.items()}
        letter_seq = ''.join([num_to_letter[i] for i in self.sequence[:len(self.sequence)-self.nb_PCA_comp]])
        print(letter_seq)
        return letter_seq

    def modify_PCA_target(self,new_PCA_comp):
        n=len(new_PCA_comp)
        if n==self.nb_PCA_comp:
            self.sequence[-self.nb_PCA_comp:]=new_PCA_comp.copy()

    def plm_calc(self, site, trial_aa):
        """
        Compute unnormalized pseudo-likelihood of trial_aa at a given site.
        site: int from 0 to L-1
        trial_aa: int from 0 to 21 (amino acid index)
        """
        if site < 0 or site >= self.L:
            raise ValueError(f"Site {site} is out of bounds for sequence length {self.L}.")
        sum_energy = 0.0
        if not ( self.J_PCA is None):
            #for i in range(self.nb_PCA_comp):
            #    sum_energy+= self.J_PCA[trial_aa,self.sequence[i],site,i]
            # PREBIOUS
            #for i in range(self.nb_PCA_comp):
            #    PCA_coord = self.sequence[self.L + i]  # the PCA coordinate at component i
            #    sum_energy += self.beta_PCA*self.J_PCA[trial_aa, PCA_coord, site, i]
            # NEW
            if self.nb_PCA_comp == 1:       # Model 3: single joint PCA bin index
                joint_bin = self.sequence[self.L]  # Already 0..1224 in sequence
                sum_energy += self.beta_PCA * self.J_PCA[trial_aa, joint_bin, site, 0]
            else:                           # Model 2: separate PCA coordinates
                for i in range(self.nb_PCA_comp):
                    PCA_coord = self.sequence[self.L + i]  # 0..34 for each coordinate
                    sum_energy += self.beta_PCA * self.J_PCA[trial_aa, PCA_coord, site, i]
        else:
            for i in range(self.nb_PCA_comp):
                sum_energy+=self.beta_PCA*self.J[trial_aa,self.sequence[self.L+i],site,self.L+i]
        for j in range(self.L):
            if j == site:
                continue
            aa_j = self.sequence[j]
            sum_energy += self.beta * self.J[trial_aa, aa_j, site, j] # check indexing
            #sum_energy += self.J[aa_j, trial_aa, j, site] 
        prob = sum_energy  
        return prob
    
    def plm_site_distribution(self, site):
        """
        Compute probability distriution for specific site (normalized)
        """
        probs = []
        for trial_aa in range(21):
            probs.append(self.plm_calc(site, trial_aa))
        probs = np.array(probs)
        probs = np.exp(probs-probs.max()) #to avoid overflow for high beta
        probs /= probs.sum()
        return probs
    
    def draw_aa(self, site):
        """
        Sample a new AA at the given site from PLM distribution
        """
        probs = self.plm_site_distribution(site)
        new_aa = np.random.choice(21, p=probs) # aa from 0 to 20
        self.sequence[site] = new_aa

    

    def seq_energy(self):
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                sum+=self.J[self.sequence[i], self.sequence[j],i,j]
        return sum
    



class BatchSequencePLM:
    def __init__(self, J, N, beta=1, nb_PCA_comp=0, PCA_component_list=None, J_tens_PCA=None, beta_PCA=1):
        """
        Initialize with a batch of N independent sequences.
        """
        self.J = J
        self.J_PCA = J_tens_PCA
        self.beta = beta
        self.beta_PCA = beta_PCA
        self.nb_PCA_comp = nb_PCA_comp
        self.N = N

        if self.J_PCA is not None:
            self.L = J.shape[-1]
        else:
            self.L = J.shape[-1] - nb_PCA_comp

        # Initialize N random sequences
        core_sequences = np.random.randint(0, 21, size=(N, self.L))

        if nb_PCA_comp > 0:
            print(len(PCA_component_list) )
            print(nb_PCA_comp)
            if PCA_component_list is None:
                raise ValueError("PCA_component_list must be provided for PCA components.")
            if len(PCA_component_list) != nb_PCA_comp:
                raise ValueError("Mismatch between number of PCA components and list provided.")
            pca_array = np.tile(PCA_component_list, (N, 1))  # replicate for each sequence
            self.sequences = np.concatenate((core_sequences, pca_array), axis=1)
        else:
            self.sequences = core_sequences  # shape: (N, L [+ nb_PCA])

    def plm_calc_batch(self, site, trial_aa):
        """
        Compute unnormalized pseudo-likelihood for all N sequences at one site and one trial AA.
        Return: (N,) array
        """
        aa_j_all = self.sequences[:, :self.L]  # shape (N, L)
        aa_trial = np.full(self.N, trial_aa)

        energy = np.zeros(self.N)

        for j in range(self.L):
            if j == site:
                continue
            aa_j = aa_j_all[:, j]  # shape (N,)
            energy += self.beta * np.asarray(self.J[trial_aa, aa_j, site, j])  # vectorized lookup

        # If PCA is used
        if self.nb_PCA_comp > 0:
            for i in range(self.nb_PCA_comp):
                PCA_coord = self.sequences[:, self.L + i]
                if self.J_PCA is not None:
                    energy += self.beta_PCA * np.asarray(self.J_PCA[trial_aa, PCA_coord, site, i])
                else:
                    energy += self.beta_PCA * np.asarray(self.J[trial_aa, PCA_coord, site, self.L + i])
        
        return energy

    def plm_site_distribution_batch(self, site):
        """
        Compute PLM probabilities for each AA (0..20) for all N sequences at site.
        Returns: (N, 21) array of probabilities
        """
        raw_scores = np.zeros((self.N, 21))
        for aa in range(21):
            raw_scores[:, aa] = self.plm_calc_batch(site, aa)

        raw_scores -= raw_scores.max(axis=1, keepdims=True)  # stability
        probs = np.exp(raw_scores)
        probs /= probs.sum(axis=1, keepdims=True)  # normalize
        return probs  # shape: (N, 21)
    
    def plm_calc_batch_diff_J(self, J,site, trial_aa):
        """
        Compute unnormalized pseudo-likelihood for all N sequences at one site and one trial AA.
        Return: (N,) array
        """
        aa_j_all = self.sequences[:, :self.L]  # shape (N, L)
        aa_trial = np.full(self.N, trial_aa)

        energy = np.zeros(self.N)

        for j in range(self.L):
            if j == site:
                continue
            aa_j = aa_j_all[:, j]  # shape (N,)
            energy += self.beta * np.asarray(J[trial_aa, aa_j, site, j])  # vectorized lookup

        # If PCA is used
        if self.nb_PCA_comp > 0:
            for i in range(self.nb_PCA_comp):
                PCA_coord = self.sequences[:, self.L + i]
                if self.J_PCA is not None:
                    energy += self.beta_PCA * np.asarray(self.J_PCA[trial_aa, PCA_coord, site, i])
                else:
                    energy += self.beta_PCA * np.asarray(J[trial_aa, PCA_coord, site, self.L + i])
        
        return energy

    def compare_J(self,J ,site):
        """
        Compute PLM probabilities for each AA (0..20) for all N sequences at site.
        Returns: (N, 21) array of probabilities
        """
        prob_list_self_J=self.plm_site_distribution_batch(site)
        raw_scores = np.zeros((self.N, 21))
        for aa in range(21):
            raw_scores[:, aa] = self.plm_calc_batch_diff_J(J,site,aa)

        raw_scores -= raw_scores.max(axis=1, keepdims=True)  # stability
        probs = np.exp(raw_scores)
        probs /= probs.sum(axis=1, keepdims=True)  # normalize
        diff=abs(prob_list_self_J-probs).mean()
        return diff  # shape: (N, 21)

    def assign_seqs(self,sequences):
        self.sequences=sequences

    def draw_aa_batch(self, site):
        """
        Sample new AAs for all N sequences at a given site.
        """
        probs = self.plm_site_distribution_batch(site)  # (N, 21)
        new_aas = np.array([np.random.choice(21, p=p) for p in probs])  # (N,)
        self.sequences[:, site] = new_aas

    def evolve_all(self, n_iter=500):
        """
        Perform Gibbs sampling over all sequences for n_iter iterations.
        """
        for _ in tqdm(range(n_iter)):
            site=np.random.choice(self.L)
            self.draw_aa_batch(site)

    def get_sequences(self):
        """
        Return all N sequences.
        """
        return self.sequences

    def to_letters(self):
        """
        Return N sequences in letter format.
        """
        num_to_letter = {v: k for k, v in letter_to_num.items()}
        letter_seqs = []
        for seq in self.sequences[:, :self.L]:
            letter_seq = ''.join([num_to_letter[i] for i in seq])
            letter_seqs.append(letter_seq)
        return letter_seqs
    


def flatten(coords, nb_bins_PCA):
    x_idx, y_idx = coords
    return x_idx * nb_bins_PCA + y_idx


