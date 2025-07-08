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
        sum_energy = 0.0
        if not ( self.J_PCA is None):
            #for i in range(self.nb_PCA_comp):
            #    sum_energy+= self.J_PCA[trial_aa,self.sequence[i],site,i]
            for i in range(self.nb_PCA_comp):
                PCA_coord = self.sequence[self.L + i]  # the PCA coordinate at component i
                sum_energy += self.J_PCA[trial_aa, PCA_coord, site, i]
        for j in range(self.L):
            if j == site:
                continue
            aa_j = self.sequence[j]
            sum_energy += self.beta * self.J[trial_aa, aa_j, site, j] # check indexing
            #sum_energy += self.J[aa_j, trial_aa, j, site] 
        prob = np.exp(sum_energy)  # unnormalized
        return prob
    
    def plm_site_distribution(self, site):
        """
        Compute probability distriution for specific site (normalized)
        """
        probs = []
        for trial_aa in range(21):
            probs.append(self.plm_calc(site, trial_aa))
        probs = np.array(probs)
        probs /= probs.sum()
        return probs
    
    def draw_aa(self, site):
        """
        Sample a new AA at the given site from PLM distribution
        """
        probs = self.plm_site_distribution(site)
        new_aa = np.random.choice(21, p=probs) # aa from 0 to 20
        self.sequence[site] = new_aa

    def update_PCA_coords(self, plot=True):
        """
        Jointly update both PCA coordinates using Boltzmann sampling
        based on PLM-derived J_PCA tensor and the current amino acid sequence.

        Assumes: nb_PCA_comp == 2
        """
        if self.J_PCA is None:
            raise ValueError("J_PCA not provided in model.")
        if self.nb_PCA_comp != 2:
            raise ValueError(f"Joint 2D PCA update only supports 2 PCA components, got {self.nb_PCA_comp}.")

        L = self.L
        Nbins = self.J_PCA.shape[1]
        offset = L  # Start index of PCA coords in self.sequence
        probs_2D = np.zeros((Nbins, Nbins))
        energies_2D = np.zeros((Nbins, Nbins))

        # Compute energy for each (i,j) PCA coordinate pair
        for i in range(Nbins):  # PCA component 0
            for j in range(Nbins):  # PCA component 1
                energy = 0.0
                for pos in range(L):
                    aa = self.sequence[pos]
                    energy += self.beta_PCA * (
                        self.J_PCA[aa, i, pos, 0] + self.J_PCA[aa, j, pos, 1]
                    )
                energies_2D[i, j] = energy  # Store energy for visualization

        # Numerically stable softmax
        shifted_energies = -self.beta_PCA * energies_2D
        #shifted_energies -= shifted_energies.max()

        probs_2D = np.exp(shifted_energies)
        probs_2D /= probs_2D.sum()

        total = probs_2D.sum()

        if total == 0 or np.isnan(total) or np.isinf(total):
            probs_2D = np.ones((Nbins, Nbins)) / (Nbins * Nbins)
        else:
            probs_2D /= total

        if plot==True:
            import matplotlib.pyplot as plt
            # Plot Energy Landscape
            plt.figure(figsize=(7, 6))
            plt.imshow(energies_2D, origin='lower', cmap='coolwarm')
            plt.colorbar(label='Energy')
            plt.title("Energy Landscape (Lower = Preferred)")
            plt.xlabel("PCA Component 1 (X-axis)")
            plt.ylabel("PCA Component 0 (Y-axis)")
            plt.xticks(np.arange(Nbins))
            plt.yticks(np.arange(Nbins))
            plt.tight_layout()
            plt.show()
            # Visualize probability surface
            plt.figure(figsize=(7, 6))
            plt.imshow(probs_2D, origin='lower', cmap='viridis')
            plt.colorbar(label='Probability')
            plt.title("Joint PCA Coordinate Probability Distribution")
            plt.xlabel("PCA Component 1 (X-axis)")
            plt.ylabel("PCA Component 0 (Y-axis)")
            plt.xticks(np.arange(Nbins))
            plt.yticks(np.arange(Nbins))
            plt.tight_layout()
            plt.show()

        # Sample from joint distribution
        flat_probs = probs_2D.flatten()
        choice = np.random.choice(Nbins * Nbins, p=flat_probs)
        i_sampled, j_sampled = np.unravel_index(choice, (Nbins, Nbins))

        self.sequence[offset + 0] = i_sampled  # PCA comp 0
        self.sequence[offset + 1] = j_sampled  # PCA comp 1

        return np.array([i_sampled, j_sampled])
    


    def seq_energy(self):
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                sum+=self.J[self.sequence[i], self.sequence[j],i,j]
        return sum