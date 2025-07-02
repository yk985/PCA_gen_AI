from tqdm import tqdm
import numpy as np

from seq_utils import letter_to_num

class SequenceMC:
    # Logic: at each step, change AA with probability p, or keep it with probability 1-p
    def __init__(self, J, initial_sequence = None, beta = 1):
        """
        Initialize the SequenceMC object with a coupling tensor J of the family and an optional initial sequence.
        """
        self.J = J
        self.L = J.shape[-1]
        self.beta = beta
        if initial_sequence is None:
            self.sequence = np.random.choice(np.arange(21), self.L) # Sequence of ints (1 to 21)
        else:
            self.sequence = initial_sequence

    def to_letter(self):
        """
        Show sequence as letters
        """
        print("Sequence:", self.sequence)
        num_to_letter = {v: k for k, v in letter_to_num.items()}
        letter_seq = ''.join([num_to_letter[i] for i in self.sequence])
        print(letter_seq)
        return letter_seq

    def site_energy(self, site):
        sum=0
        for i in range(self.L):
            if i != site:
                sum+=self.J[self.sequence[site], self.sequence[i],site,i]
        return sum*(-1)

    def draw_aa_metropolis(self, site):
        """
        Sample a new AA at the given site from PLM distribution
        """
        current_aa = self.sequence[site]
        trial_aa = np.random.choice([aa for aa in range(21) if aa != current_aa])

        # Compute current and proposed energy (can optimize to diff only affected terms)
        E_old = self.site_energy(site)
        self.sequence[site] = trial_aa
        E_new = self.site_energy(site)

        # Restore the original amino acid
        self.sequence[site] = current_aa

        # Compute the acceptance probability
        delta_E = E_new - E_old
        accept_prob = min(1, np.exp(-self.beta * delta_E))  # Use min(1, exp(-βΔE))

        if np.random.rand() < accept_prob:
            self.sequence[site] = trial_aa  # Accept the new amino acid

    def seq_energy(self):
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                sum+=self.J[self.sequence[i], self.sequence[j],i,j]
        return sum