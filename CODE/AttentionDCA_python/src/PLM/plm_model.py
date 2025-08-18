import site
from tqdm import tqdm
import numpy as np

from seq_utils import letter_to_num

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

    def update_PCA_coords(self, model, plot=False):
        """
        Jointly update both PCA coordinates using Boltzmann sampling
        based on PLM-derived J_PCA tensor and the current amino acid sequence.

        Assumes: nb_PCA_comp == 2
        """
        #if self.J_PCA is None:
        #    raise ValueError("J_PCA not provided in model.")
        #if self.nb_PCA_comp != 2:
        #    raise ValueError(f"Joint 2D PCA update only supports 2 PCA components, got {self.nb_PCA_comp}.")

        L = self.L
        Nbins = 35
        offset = L  # Start index of PCA coords in self.sequence
        probs_2D = np.zeros((Nbins, Nbins))
        energies_2D = self.compute_coord_energy(model)

        # Numerically stable softmax
        shifted_energies = self.beta_PCA * energies_2D #(energy should be -)
        #shifted_energies -= shifted_energies.max()

        shifted_energies -= shifted_energies.max()  # subtract max for numerical stability

        probs_2D = np.exp(-shifted_energies)
        total = probs_2D.sum()

        # Check for numerical issues
        if total == 0 or np.isnan(total) or np.isinf(total):
            probs_2D = np.ones_like(probs_2D) / probs_2D.size  # uniform distribution
        else:
            probs_2D /= total

        # Sample from joint distribution
        flat_probs = probs_2D.flatten()
        choice = np.random.choice(Nbins * Nbins, p=flat_probs)
        i_sampled, j_sampled = np.unravel_index(choice, (Nbins, Nbins))

        self.sequence[offset + 0] = i_sampled  # PCA comp 0
        self.sequence[offset + 1] = j_sampled  # PCA comp 1

        return np.array([i_sampled, j_sampled])
    
    def compute_coord_energy(self, model):
        """
        model = 1,2,3
        """
        L = self.L
        Nbins = 35
        offset = L 
        energies_2D = np.zeros((Nbins, Nbins))
        energies_flat = np.zeros((Nbins*Nbins))
        if model == 1:
            # for i in range(Nbins):  # PCA comp 0
            #     for j in range(Nbins):  # PCA comp 1
            #         energy = 0.0
            #         for pos in range(L):  # real amino acid sites
            #             aa = self.sequence[pos]  # actual residue index (0–20)
            #             # Interact with PCA comp 0 (stored at position L)
            #             pca_aa0 = i  # interpreted as "amino acid index" at pos = L
            #             energy += self.J_PCA[aa, pca_aa0, pos, 0]

            #             # Interact with PCA comp 1 (stored at position L+1)
            #             pca_aa1 = j
            #             energy += self.J_PCA[aa, pca_aa1, pos, 1]
            # # Scale by beta_PCA (if used only for PCA couplings)
            #         energies_2D[i, j] = self.beta_PCA * energy
            ##Comment j'aurais fait
            for i in range(Nbins):  # PCA comp 0
                for j in range(Nbins):  # PCA comp 1
                    energy = 0.0
                    self.sequence[-2]=i
                    self.sequence[-1]=j
                    for pos in range(L+self.nb_PCA_comp):  # real amino acid sites
                        aa = self.sequence[pos]  # actual residue index (0–20)
                        # Interact with PCA comp 0 (stored at position L)
                        pca_aa0 = i  # interpreted as "amino acid index" at pos = L
                        if pos!=L :
                            energy += self.J[aa, pca_aa0, pos, -2]

                        # Interact with PCA comp 1 (stored at position L+1)
                        pca_aa1 = j
                        if pos != L+1:
                            energy += self.J[aa, pca_aa1, pos, -1]

            # Scale by beta_PCA (if used only for PCA couplings)
                    energies_2D[i, j] = self.beta_PCA * energy

        elif model == 2:
        # Compute energy for each (i,j) PCA coordinate pair
            for i in range(Nbins):  # PCA component 0
                for j in range(Nbins):  # PCA component 1
                    energy = 0.0
                    for pos in range(L):
                        aa = self.sequence[pos]
                        energy += self.beta_PCA * (self.J_PCA[aa, i, pos, 0] + self.J_PCA[aa, j, pos, 1])
                    energies_2D[i, j] = energy  # Store energy for visualization

        elif model == 3:
            for i in range(Nbins):
                energy = 0.0
                for pos in range(L):
                    aa = self.sequence[pos]
                    energy += self.beta_PCA * (self.J_PCA[aa, i, pos])
                energies_flat[i] = energy.item()
            energies_2D = energies_flat.reshape((Nbins, Nbins))
        energies_2D = -energies_2D  # Negate to match Boltzmann distribution (lower energy = higher probability)
        return energies_2D

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
            self.L = J.shape[-1] - nb_PCA_comp
        else:
            self.L = J.shape[-1] - nb_PCA_comp

        # Initialize N random sequences
        core_sequences = np.random.randint(0, 21, size=(N, self.L))

        if nb_PCA_comp > 0:
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


class SequenceCondPLM:
    def __init__(self, J, G, initial_sequence = None, beta = 1, nb_PCA_comp=0,PCA_component_list=np.array([]),beta_PCA=1):
        """
        Initialize the SequencePLM object with a coupling tensor J of the family and an optional initial sequence.
        """
        self.J = J # shape (q,q,L,L)
        self.G = G # shape (L,q,Nbins*Nbins)
        #self.L = J.shape[-1]
        self.beta = beta
        self.beta_PCA=beta_PCA
        self.nb_PCA_comp=nb_PCA_comp
        self.L = J.shape[-1]
        self.Nbins = np.pow(G.shape[-1], 1/nb_PCA_comp).astype(int) if nb_PCA_comp > 0 else 1  # Number of bins for PCA components
        if initial_sequence is None:
            np.random.seed(42)
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
        self.PCA_coord = np.array(PCA_component_list) if PCA_component_list.size > 0 else np.zeros(nb_PCA_comp)

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
        for j in range(self.L):
            if j == site:
                continue
            aa_j = self.sequence[j]
            sum_energy += self.beta * self.J[trial_aa, aa_j, site, j] # check indexing
            #sum_energy += self.J[aa_j, trial_aa, j, site] 
        if self.nb_PCA_comp > 0:
            flat_coord = flatten(self.PCA_coord, self.Nbins).astype(int)  # Flatten PCA coordinates to a single index
            if flat_coord < 0 or flat_coord >= self.G.shape[-1]:
                raise ValueError(f"Flat coordinate {flat_coord} is out of bounds for G tensor with shape {self.G.shape}.")
            sum_energy += self.beta_PCA * self.G[site, trial_aa, flat_coord]  # Add contribution from G tensor
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