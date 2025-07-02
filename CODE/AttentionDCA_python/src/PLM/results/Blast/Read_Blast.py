import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns  
import matplotlib 

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
font = {'size'   : 18}

matplotlib.rc('font', **font)

# Load your BLAST result file
#filename = "gill_100seqs_randinit.txt"
#filename = "gill_100seqs_initseq.txt"
#filename = "mc_100seqs_randinit.txt"
#filename = "mc_100seqs_initseq.txt"
#filename = "plm_100seqs_randinit.txt"
filename = "plm_100seqs_initseq.txt"
df = pd.read_csv(filename, sep='\t', comment='#', header=None)

# Assign column names
df.columns = [
    "query", "subject", "identity", "alignment_length", "mismatches", "gap_opens",
    "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score", "positives"
]

# Get the best match for each query (highest identity or lowest evalue)
best_hits = df.sort_values(by=["query", "identity"], ascending=[True, False]).groupby("query").first()

# Display
print(best_hits[["subject", "identity", "evalue", "bit_score"]])

### OLD
# Extract the arrays for identity, alignment length, and bit score
#identity_array = df['identity'].to_numpy()
#alignment_length_array = df['alignment_length'].to_numpy()
#bit_score_array = df['bit_score'].to_numpy()
#
### NEW: only best hits
best_hits = df.sort_values(by=["query", "identity"], ascending=[True, False]).groupby("query").first()
identity_array = best_hits['identity'].to_numpy()
alignment_length_array = best_hits['alignment_length'].to_numpy()
bit_score_array = best_hits['bit_score'].to_numpy()


print(np.shape(identity_array))
print(np.shape(alignment_length_array))
print(np.shape(bit_score_array))

#####

# Compute basic statistics
identity_mean = np.mean(identity_array)
identity_std = np.std(identity_array)

alignment_length_mean = np.mean(alignment_length_array)
alignment_length_std = np.std(alignment_length_array)

bit_score_mean = np.mean(bit_score_array)
bit_score_std = np.std(bit_score_array)

# Print the statistics
print(f"Identity - Mean: {identity_mean:.2f}, Std Dev: {identity_std:.2f}")
print(f"Alignment Length - Mean: {alignment_length_mean:.2f}, Std Dev: {alignment_length_std:.2f}")
print(f"Bit Score - Mean: {bit_score_mean:.2f}, Std Dev: {bit_score_std:.2f}")

def describe(name, array):
    print(f"{name} Summary:")
    print(f"  Median: {np.median(array):.2f}")
    print(f"  25th Percentile (Q1): {np.percentile(array, 25):.2f}")
    print(f"  75th Percentile (Q3): {np.percentile(array, 75):.2f}")
    print(f"  Min: {np.min(array):.2f}")
    print(f"  Max: {np.max(array):.2f}")
    print(f"  Mean: {np.mean(array):.2f}")
    print(f"  Std Dev: {np.std(array):.2f}")
    print()

describe("Identity", identity_array)
describe("Alignment Length", alignment_length_array)
describe("Bit Score", bit_score_array)

save_dir = os.path.join(os.getcwd(), "blast_plots")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


from scipy.stats import norm

# Set a nicer style
sns.set_style("whitegrid")
plt.figure(figsize=(18, 5))

# Identity Histogram
plt.subplot(1, 2, 1)
count, bins, ignored = plt.hist(identity_array, bins=20, color='#4C72B0', edgecolor='black', alpha=0.85, density=True)
# Fit a normal distribution
mu, std = norm.fit(identity_array)
# Plot the PDF (Probability Density Function)
x = np.linspace(min(identity_array), max(identity_array), 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian Fit\nμ={mu:.2f}, σ={std:.2f}')
# Labels and formatting
#plt.title("Percent Identity Distribution")
plt.xlabel("Percent Identity")
plt.ylabel("Frequency")
plt.legend()
#plt.hist(identity_array, bins=20, color='#4C72B0', edgecolor='black', alpha=0.85)

## Alignment Length Histogram
#plt.subplot(1, 3, 2)
#plt.hist(alignment_length_array, bins=20, color='#4C72B0', edgecolor='black', alpha=0.85)
#plt.title("Alignment Length Distribution")
#plt.xlabel("Alignment Length")
#plt.ylabel("Frequency")
#plt.grid(True)

plt.subplot(1, 2, 2)
# Histogram with density for overlay
count, bins, ignored = plt.hist(bit_score_array, bins=20, color='#4C72B0', edgecolor='black', alpha=0.85, density=True)
# Fit Gaussian
mu_b, std_b = norm.fit(bit_score_array)
x_b = np.linspace(min(bit_score_array), max(bit_score_array), 1000)
pdf_b = norm.pdf(x_b, mu_b, std_b)
plt.plot(x_b, pdf_b, 'r--', linewidth=2, label=f'Gaussian Fit\nμ={mu_b:.2f}, σ={std_b:.2f}')
#plt.title("Bit Score Distribution")
plt.xlabel("Bit Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.tight_layout()
save_name = f"blast_histograms_{filename.split('.')[0]}"
plt.savefig(os.path.join(save_dir, save_name), dpi=300)
plt.show()