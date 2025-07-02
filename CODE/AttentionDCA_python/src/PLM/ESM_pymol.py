import requests
import os
import subprocess
import tempfile
import time
import numpy as np
import random
import textwrap
# Provided mapping from letters to numbers
letter_to_num_conversion = {
    'A': 0,  'C': 1,  'D': 2,  'E': 3,
    'F': 4,  'G': 5,  'H': 6,  'I': 7,  
    'K': 8,  'L': 9,  'M': 10, 'N': 11, 
    'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19,
    '-': 20  # Gap symbol
}

# Invert number-to-letter (some numbers may map to multiple letters if reused)
def invert_dict(d):
    inverted = {}
    for letter, num in d.items():
        if num not in inverted:
            inverted[num] = []
        inverted[num].append(letter)
    return inverted

# Inverted map: {0: ['A'], 1: ['C'], ..., 20: ['-']}
num_to_letter_map = invert_dict(letter_to_num_conversion)

# Convert one sequence of numbers to letters
def numbers_to_letters(numbers, inverted_dict=num_to_letter_map):
    """
    Convert a list or NumPy array of amino acid numbers to a string of letters.
    """
    result = ''
    for number in numbers:
        if number in inverted_dict:
            letters = inverted_dict[number]
            result += random.choice(letters)  # In case of multiple letters (e.g. redundancy)
        else:
            raise ValueError(f"[Error] Number {number} not found in amino acid mapping.")
    return result

# Optional: Convert a batch of sequences (2D array)
def batch_numbers_to_letters(sequence_array):
    """
    Converts a 2D array of numerical protein sequences to a list of letter sequences.
    """
    return [numbers_to_letters(seq) for seq in sequence_array]

def fetch_structure(sequence, filename, retries=3, delay=5):
    """Send a sequence to ESMFold API and save the returned PDB structure with retries."""
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    sequence = str(sequence).upper()
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, data=sequence, headers={"Content-Type": "text/plain"}, timeout=60)
            if response.status_code == 200:
                with open(filename, 'w') as f:
                    f.write(response.text)
                print(f"✅ Saved PDB to: {filename}")
                return
            else:
                print(f"[Attempt {attempt}] Server error {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            print(f"[Attempt {attempt}] Request timed out.")
        
        if attempt < retries:
            print(f"⏳ Retrying in {delay} seconds...")
            time.sleep(delay)

    raise Exception("❌ Failed to fetch structure after multiple attempts.")

def filter_valid_sequences_np(np_sequences):
    """
    Filters out invalid protein sequences from a NumPy array.
    Only sequences containing the 20 standard amino acids are kept.
    Returns a filtered NumPy array of valid sequences.
    """
    valid_aa = set("ARNDCEQGHILKMFPSTWYV")
    valid_list = []

    for seq in np_sequences:
        seq_str = str(seq).upper()
        if set(seq_str).issubset(valid_aa):
            valid_list.append(seq_str)
        #else:
            #print(f"[Removed] Invalid sequence: {seq_str}")

    filtered_array = np.array(valid_list)
    print(f"\n✅ Valid sequences kept: {len(filtered_array)} / {len(np_sequences)}")
    return filtered_array

def calculate_rmsd_pymol(pdb1, pdb2):
    """Use PyMOL to align two structures and return the RMSD."""
    script = textwrap.dedent(f"""
    load {pdb1}, mol1
    load {pdb2}, mol2
    python
    from pymol import cmd
    r = cmd.align("mol1", "mol2")
    print("CUSTOM_RMSD:", r[0])
    python end
    quit
""")
    
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as script_file:
        script_file.write(script)
        script_path = script_file.name

    result = subprocess.run(["pymol", "-cq", script_path], capture_output=True, text=True)
    os.unlink(script_path)
    # print("=== PYMOL OUTPUT ===")
    # print(result.stdout)
    for line in result.stdout.splitlines():
        if "CUSTOM_RMSD:" in line:
            try:
                # Extract only the part after 'CUSTOM_RMSD:' and strip it
                value_part = line.split("CUSTOM_RMSD:")[1].strip()
                # Optional: remove any trailing characters or garbage
                value_part = value_part.split()[0]  # Takes only the first token
                rmsd_value = float(value_part)
                return rmsd_value
            except (IndexError, ValueError) as e:
                print(f"⚠️ Failed to parse RMSD line: {line} — {e}")

    raise RuntimeError("Failed to extract RMSD from PyMOL output.")

def compare_sequence_sets(true_sequences, generated_sequences,lab1="",lab2=""):
    """Compare two arrays: true vs generated. Returns pairwise RMSDs."""
    true_pdbs = []
    gen_pdbs = []
    print(len(true_sequences))
    print(len(generated_sequences))
    rmsd_matrix=np.zeros((len(true_sequences),len(generated_sequences)))
    print(rmsd_matrix.shape)
    # Predict and save structures for true sequences
    for i, seq in enumerate(true_sequences):
        filename = lab1+f"_{i+1}.pdb"
        if not os.path.exists(filename):
            print(f"✅ File {filename} created —  fetched.")
            fetch_structure(seq, filename)
            time.sleep(1)
        true_pdbs.append(filename)
        

    # Predict and save structures for generated sequences
    for j, seq in enumerate(generated_sequences):
        filename = lab2+f"_{j+1}.pdb"
        if not os.path.exists(filename):
            print(f"✅ File {filename} created —  fetched.")
            fetch_structure(seq, filename)
            time.sleep(1)
        gen_pdbs.append(filename)
    # Pairwise RMSD comparison
    
    for i, pdb_true in enumerate(true_pdbs):
        for j, pdb_gen in enumerate(gen_pdbs):
            rmsd = calculate_rmsd_pymol(pdb_true, pdb_gen)
            rmsd_matrix[i][j] = rmsd
            #print(f"True {i+1} vs Generated {j+1}: RMSD = {rmsd:.3f} Å")

    return rmsd_matrix

