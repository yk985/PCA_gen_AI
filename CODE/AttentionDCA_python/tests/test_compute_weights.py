import os
from CODE.AttentionDCA_python.src.utils import ReadFasta,read_fasta_alignment, remove_duplicate_sequences
import numpy as np


from julia import Julia

# Initialize Julia
jl = Julia(compiled_modules=False)  # Set compiled_modules=True if you face issues



from julia import Main, DCAUtils

# Import the necessary Julia packages
Main.using("DCAUtils")
Main.using("PottsGauge")



def _compute_weights(Z, theta, verbose=True):
    from julia import Julia,  DCAUtils, Main, Base


    # Convert Z to Int8
    Z_int8 = Z.astype(np.int8)
    Z_int8 = np.ascontiguousarray(Z_int8)

    
     # Ensures θ is a Julia Symbol
    new_var = jl.eval(f"PyCall.pyjlwrap_new(:{theta})")

    #theta = Main.eval(":auto")

    # Verify θ is a Julia Symbol
    print(f"Type of θ: {type(theta)}")  # Should output <class 'julia.core.Symbol'>

    # Determine q
    q = int(Z.max()) + 1  # Assuming Z contains values from 0 to q-1

    # Call compute_weights
    W, Meff = DCAUtils.compute_weights(Z_int8, q, new_var, verbose=True)

    return W, Meff






def test_compute_weights():


    cwd = os.getcwd()
    fastafile = cwd + '/CODE/DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz'    
    
    W, Zint, N, M, _ = ReadFasta(fastafile, 0.9, 'auto', True, verbose=False)
    
    remove_dups = True
    verbose = True
    max_gap_fraction = 0.9
    cwd = os.getcwd()    
    Z = read_fasta_alignment(fastafile, max_gap_fraction)
    if remove_dups:
        Z, _ = remove_duplicate_sequences(Z, verbose=verbose)
    N, M = Z.shape
    q = int(round(Z.max()))
    if q > 32:
        raise ValueError(f"Parameter q={q} is too big (max 31 is allowed)")
    #theta = 0.33088461264010915
    theta = 'auto'
    W_test, Meff_test = _compute_weights(Z, theta= theta, verbose=False)

    W_test /= Meff_test  # Normalize W

    np.testing.assert_allclose(W, W_test,  rtol=1e-10, atol=0)


if __name__ == '__main__':
    test_compute_weights()
    print("test passed")




    