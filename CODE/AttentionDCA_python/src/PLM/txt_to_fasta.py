from seq_utils import nums_to_letters
import numpy as np
# Read your sequences from a txt file and output a fasta file

seq_dir = 'mc_generated_sequences'
#filename = 'generated_sequences_randinit_40000'
#filename = 'mc_gen_seqs_w_init_seq_Ns300000_r0.1'

seq_dir = 'generated_sequences'
#filename = 'generated_sequences_randinit_40000'
#filename = 'gen_seqs_w_init_seq_Ns40000_r0.1'

seq_dir = 'gill_generated_sequences'
#filename = 'gill_gen_seqs_w_init_seq_Ns30000_r0.3_after_mod'
filename = 'gill_generated_sequences_randinit_20000'

seq_dir = 'generated_sequences_brute_forcePCA_comp'
filename = 'gen_seqs_randinit_Ns15000_b_1_b_PCA10_PCA_comp_24_20PCA_comp'

output_file = f'{seq_dir}/{filename}.npy'
gill_gen_seqs = np.load(output_file)

## convert sequences to a list of strings using nums_to_letters
#gill_gen_seqs = [nums_to_letters(seq) for seq in gill_gen_seqs]
## Save the sequences to a text file
#with open(f'{seq_dir}/{filename}.txt', 'w') as outfile:
#    for seq in gill_gen_seqs:
#        outfile.write(f"{seq}\n")


txt_file = f'{seq_dir}/{filename}.txt'

# Read all sequences first
with open(txt_file, 'r') as infile:
    sequences = [line.strip().replace('-', '') for line in infile if line.strip()]

# Make sure there are enough sequences
if len(sequences) <= 5000:
    raise ValueError("Not enough sequences to start at index 5000!")
# Check size of a sequence
if len(sequences[0]) != 63:
    raise ValueError("Sequences are not of length 63, please check the input file.")

# Select sequences starting from index 5000
selected_sequences = sequences[5000:]

# Pick 100 evenly spaced sequences
import numpy as np
indices = np.linspace(0, len(selected_sequences) - 1, 100, dtype=int)
evenly_spaced_sequences = [selected_sequences[i] for i in indices]

# Now write to fasta
with open(f'{seq_dir}/{filename}.fasta', 'w') as outfile:
    for idx, seq in enumerate(evenly_spaced_sequences):
        outfile.write(f">seq{5000 + indices[idx] + 1}\n{seq}\n")  # adjust seq id based on original file