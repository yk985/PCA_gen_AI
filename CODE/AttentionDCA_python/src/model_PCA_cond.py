import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ModelPCAcondJ(nn.Module):
    def __init__(
        self, 
        H, 
        d, 
        N, 
        q, 
        Nbins=35,
        Ncomp=2,
        Q=None,
        V=None,
        K=None,
        lambd=0.001, 
        index_last_domain1=0, 
        H1=0, 
        H2=0, 
        init_fun=np.random.randn,
        device = 'cpu'
        
    ):
        super(ModelPCAcondJ, self).__init__()

        # Device & dtype
        # You could choose your own logic for picking device, or force CPU/GPU.
        # For example:
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")  
        self.dtype = torch.float32  # or torch.float64, etc.

        # Store hyperparameters
        self.H = H
        self.d = d
        self.N = N
        self.q = q
        self.Nbins = Nbins
        self.Ncomp = Ncomp
        self.lambd = lambd
        self.index_last_domain1 = index_last_domain1
        self.H1 = H1
        self.H2 = H2
        self.device = device
        self.Q=Q
        self.V=V
        self.K=K
        seed=0
        import random
        import numpy as np
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Define parameters
        if Q==None:
            self.Q = nn.Parameter(
                torch.tensor(np.random.randn(H, d, N), dtype=self.dtype, device=self.device)
            )
        if K==None:
            self.K = nn.Parameter(
                torch.tensor(np.random.randn(H, d, N+Ncomp), dtype=self.dtype, device=self.device)
            )
        if V==None:
            self.V = nn.Parameter(
                torch.tensor(np.random.randn(H, q, Nbins), dtype=self.dtype, device=self.device)
            )
        import os
        cwd = os.getcwd()


        def read_tensor_from_txt(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()

            # Read the dimensions from the first line
            dims = list(map(int, lines[0].strip().split()))

            # Initialize a list to hold the tensor data
            tensor_data = []

            current_slice = []
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("Slice"):
                    if current_slice:  # If there is an existing slice, save it
                        tensor_data.append(current_slice)
                        current_slice = []
                elif line:  # Process non-empty lines
                    current_slice.append(list(map(float, line.split(','))))

            if current_slice:  # Append the last slice
                tensor_data.append(current_slice)

            # Convert the list back into a tensor with the original dimensions
            tensor = torch.tensor(tensor_data).view(*dims)
            return tensor
        

        # K = read_tensor_from_txt( cwd +"/results/34_23_save_random_without_J_400/K_tensor.txt")
        # Q= read_tensor_from_txt( cwd +"/results/34_23_save_random_without_J_400/Q_tensor.txt")
        # V = read_tensor_from_txt( cwd +"/results/34_23_save_random_without_J_400/V_tensor.txt")



        # self.Q.data = Q
        # self.K.data = K
        # self.V.data = V

    def forward(self, Z, weights):
        # Forward just calls the loss, as in your original code
        loss_value = self.loss_wo_J(
            self.Q, 
            self.K, 
            self.V, 
            Z, 
            weights,
            lambd=self.lambd,
            index_last_domain1=self.index_last_domain1,
            H1=self.H1,
            H2=self.H2
        )
        return loss_value

    ###########################################################################
    #                            Helper Methods                                #
    ###########################################################################

    def compute_product_Q_K(self, Q, K):
        
        device = self.device
        dtype = self.dtype

        H, _, N = Q.shape
        H, _, M = K.shape #(M=N+Nbins)
        # Step 1: Compute the raw attention scores using einsum
        e = torch.einsum('hdi,hdj->ijh', Q, K)  # Shape: (N, M, H)

        # Exclude self-interactions by setting scores to -inf on the diagonal
        i_indices = torch.arange(N, device=device).unsqueeze(1)
        j_indices = torch.arange(N, device=device).unsqueeze(0)
        self_mask = (i_indices != j_indices).float()
        # Extend mask to cover PCA positions (keep last Ncomp columns unmasked)
        if M > N:
            ones_for_pca = torch.ones(N, M - N, device=device)       # (N, Ncomp)
            mask_extended = torch.cat([self_mask, ones_for_pca], dim=1)  # (N, M)
        else:
            mask_extended = self_mask
        mask_value = -1e9  # A large negative value to zero out after softmax
        e = e * mask_extended.unsqueeze(-1) + (1 - mask_extended.unsqueeze(-1)) * mask_value

        # If there's a domain split:
        if self.index_last_domain1 > 0 and self.index_last_domain1 < N:
            print("Oops deleted domain split")
        else:
            domain_masks = 0
        return e

    def compute_attention_heads(self, Q, K, V, index_last_domain1=0, H1=0, H2=0):
        
        device = self.device
        dtype = self.dtype

        H, _, N = Q.shape
        H, _, M = K.shape  #(M=N+Nbins)
        # N, _, _ = V.shape  # Actually your code re-assigns same N but let's keep it as is
        _N, _, _ = V.shape
        index_first_domain2 = index_last_domain1 + 1

        # Get e from compute_product_Q_K
        e = self.compute_product_Q_K(Q, K) # shape: (N, M, H)

        sf = torch.zeros(N, M, H, device=device, dtype=dtype)
        # e has shape (N, M, H)
        sf = torch.softmax(e, dim=1)  # shape (N, M, H), softmax applied over M for each head
        #for h in range(H):
        #    if index_last_domain1 != 0:
        #        print("Oops deleted domain masks")
        #    else:
        #        # No domain masks applied
        #        sf_domain = torch.softmax(e[:, :, h], dim=1)
        #        sf = sf.clone()
        #        sf[:, :, h] = sf_domain
        return sf

    def compute_mat_ene(self, Q, K, V, Z, H1=0, H2=0, index_last_domain1=0):
        # We'll assume you intended to call self.compute_attention_heads here.
        # The old snippet references 'e' out of nowhere, so presumably that was from compute_product_Q_K.
        # We'll keep the lines exactly the same, except we clarify how 'sf' is obtained.

        # The code below uses variables that appear in the snippet,
        # but to keep it consistent, let's define them in place:
        device = self.device
        dtype = self.dtype

        H, q, _ = V.shape
        # For clarity in your snippet, 'e' was from compute_product_Q_K,
        # and 'sf' is from compute_attention_heads.
        # We'll compute them inside to match your logic:

        sf = self.compute_attention_heads(
            Q=Q, 
            K=K, 
            V=V, 
            index_last_domain1=index_last_domain1, 
            H1=H1, 
            H2=H2
        )
        # From your snippet, you used e.shape for N_e1, N_e2, H_e,
        # but actually let's just read from sf itself.
        N, N_plus_Ncomp, H_e = sf.shape
        N_seq, M = Z.shape  # Z: (N+Ncomp, M)
        Ncomp = N_plus_Ncomp - N
        assert N_plus_Ncomp == Z.shape[0], "Mismatch in key dimension"

        mat_ene = torch.zeros(N, q, M, device=device, dtype=dtype)
        # Weighted sum loop
        for h in range(H):
            V_h = V[h] # shape: V (H,q, Nbins) so V_h has shape (q, Nbins)
            # The next line in your snippet references V_h[:, Z], 
            # but that can be tricky because Z is shape (N+Ncomp, M).
            # We keep it as it is in your snippet, trusting you have reason:
            V_h_Zj = V_h[:, Z]     # shape => (q, N+Ncomp, M)
            V_h_Zj = V_h_Zj.permute(1, 0, 2)  # => (N+Ncomp, q, M)
            mat_ene_h = torch.einsum('ij,jqm->iqm', sf[:, :, h], V_h_Zj)
            mat_ene += mat_ene_h
        mat_ene = mat_ene.permute(1, 0, 2)
        # mat_ene now has shape (q, N, M)
        return mat_ene, sf

    def loss_wo_J(self, Q, K, V, Z, weights, lambd=0.001, index_last_domain1=0, H1=0, H2=0):
        device = self.device
        dtype = self.dtype

        H, d, N = Q.shape
        q = V.shape[1]  # Number of amino acids
        M = Z.shape[1]
        _, _, N_plus_Ncomp = K.shape  # K shape: (H, d, N+Ncomp)
        
        # Step: compute mat_ene and sf
        mat_ene, sf = self.compute_mat_ene(
            Q, 
            K, 
            V, 
            Z, 
            H1=H1, 
            H2=H2, 
            index_last_domain1=index_last_domain1
        )  # Shape: (q, N, M)
        # logsumexp
        lge = torch.logsumexp(mat_ene, dim=0)  # Shape: (N, M)
        Z_old = Z.clone()  # Keep original Z for gathering
        Z_old = Z_old[:N]  # Ensure Z is of shape (N, M) for gathering
        Z_indices = Z_old.unsqueeze(0)  # Shape: (1, N, M)
        mat_ene_selected = torch.gather(mat_ene, dim=0, index=Z_indices).squeeze(0)  # (N+, M)

        pl_elements = weights * (mat_ene_selected - lge) #weighted sum along M
        pl = -torch.sum(pl_elements) # sum along N

        # For the regularization term, your snippet references M_matrix, etc.
        # That part of your snippet uses `self_mask`, but it was never fully spelled out. 
        # We'll keep it exactly as in your snippet:

        # The snippet tries: M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, self_mask)
        # But 'self_mask' is not defined in this scope. If that was part of your code, 
        # you must define it. We'll keep the line as is (though it may error if `self_mask` is missing).
    # Compute regularization term
        i_indices = torch.arange(N, device=device).unsqueeze(1)
        j_indices = torch.arange(N, device=device).unsqueeze(0)
        self_mask = (i_indices != j_indices).float()  # (N, N)
        if N_plus_Ncomp > N:
            # Ones for PCA/complement coordinates
            ones_for_pca = torch.ones(N, N_plus_Ncomp - N, device=device)  # (N, Ncomp)
            # Concatenate along second dim to cover all positions
            mask_extended = torch.cat([self_mask, ones_for_pca], dim=1)  # (N, N+Ncomp)
        else:
            mask_extended = self_mask  # (N, N) if no extra coords
        M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, mask_extended)  # Shape: (H, H)
        VV = V.view(H, -1)  # Shape: (H, q*q)
        VV_T = VV @ VV.T  # Shape: (H, H)
        sum_J_squared = torch.sum(M_matrix * VV_T)  # Scalar
        reg = lambd * sum_J_squared  # Scalar

        loss_value = pl + reg

        del sf, mat_ene, mat_ene_selected, M, VV_T
        torch.cuda.empty_cache()
    
        return loss_value

class ModelPCACond(nn.Module):
    def __init__(
        self, 
        H, 
        d, 
        N, 
        q, 
        Nbins,
        Q=None,
        V=None,
        K=None,
        G=None,
        lambd=0.001, 
        index_last_domain1=0, 
        H1=0, 
        H2=0, 
        init_fun=np.random.randn,
        device = 'cpu'
    ):
        super(ModelPCACond, self).__init__()

        # Device & dtype
        # You could choose your own logic for picking device, or force CPU/GPU.
        # For example:
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")  
        self.dtype = torch.float32  # or torch.float64, etc.

        # Store hyperparameters
        self.H = H
        self.d = d
        self.N = N
        self.q = q
        self.Nbins = Nbins
        self.lambd = lambd
        self.index_last_domain1 = index_last_domain1
        self.H1 = H1
        self.H2 = H2
        self.device = device
        self.Q=Q
        self.V=V
        self.K=K
        self.G=G
        seed=0
        import random
        import numpy as np
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Define parameters
        if Q==None:
            self.Q = nn.Parameter(
                torch.tensor(np.random.randn(H, d, N), dtype=self.dtype, device=self.device)
            )
        if K==None:
            self.K = nn.Parameter(
                torch.tensor(np.random.randn(H, d, N), dtype=self.dtype, device=self.device)
            )
        if V==None:
            self.V = nn.Parameter(
                torch.tensor(np.random.randn(H, q, q), dtype=self.dtype, device=self.device)
            )
        import os
        cwd = os.getcwd()
        # PCA conditioning -> (N, q) bias
        pca_dim = 2      # number of PCA coords per sequence
        hidden_dim = 32   # you can tune this

        self.G = nn.Parameter(torch.randn(N, q, Nbins*Nbins, device=self.device, dtype=self.dtype))

        # alternative - more flexibel (?)
        #self.cond_mlp = nn.Sequential(
        #    nn.Linear(pca_dim, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, N * q)  # final output: flattened bias for N positions × q amino acids
        #).to(self.device)


        def read_tensor_from_txt(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()

            # Read the dimensions from the first line
            dims = list(map(int, lines[0].strip().split()))

            # Initialize a list to hold the tensor data
            tensor_data = []

            current_slice = []
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("Slice"):
                    if current_slice:  # If there is an existing slice, save it
                        tensor_data.append(current_slice)
                        current_slice = []
                elif line:  # Process non-empty lines
                    current_slice.append(list(map(float, line.split(','))))

            if current_slice:  # Append the last slice
                tensor_data.append(current_slice)

            # Convert the list back into a tensor with the original dimensions
            tensor = torch.tensor(tensor_data).view(*dims)
            return tensor
        


    def forward(self, Z, weights, P, g_scale=1.0):
        # Forward just calls the loss, as in your original code
        loss_value = self.loss_cond(
            self.Q,
            self.K,
            self.V,
            self.G,
            Z,
            weights,
            P,
            g_scale=g_scale,
            lambd=self.lambd,
            index_last_domain1=self.index_last_domain1,
            H1=self.H1,
            H2=self.H2
        )
        return loss_value

    ###########################################################################
    #                            Helper Methods                                #
    ###########################################################################

    def create_attention_masks(self, H, L, index_last_domain1, H1, H2):

        device = self.device
        # Initialize masks tensor
        index_first_domain2 = index_last_domain1 + 1
        masks = torch.zeros(H, L, L, device=device)

        # Define head indices for each type
        H_total = H

        # Create position indices
        positions = torch.arange(L, device=device)

        # Create domain masks
        domain1_mask = (
            (positions.unsqueeze(1) <= index_last_domain1) 
            & (positions.unsqueeze(0) <= index_last_domain1)
        )
        domain2_mask = (
            (positions.unsqueeze(1) >= index_first_domain2) 
            & (positions.unsqueeze(0) >= index_first_domain2)
        )
        inter_domain_mask = (
            ((positions.unsqueeze(1) <= index_last_domain1) 
             & (positions.unsqueeze(0) >= index_first_domain2))
            | ((positions.unsqueeze(1) >= index_first_domain2) 
               & (positions.unsqueeze(0) <= index_last_domain1))
        )

        # Assign masks to heads
        for h in range(H_total):
            if h < H1:
                # Heads for Domain 1
                masks[h] = domain1_mask.float()
            elif h < H2:
                # Heads for Domain 2
                masks[h] = domain2_mask.float()
            else:
                # Heads for inter-domain interactions
                masks[h] = inter_domain_mask.float()

        return masks

    def compute_product_Q_K(self, Q, K):
        device = self.device
        dtype = self.dtype
        H, _, N = Q.shape
        # Step 1: Compute the raw attention scores using einsum
        e = torch.einsum('hdi,hdj->ijh', Q, K)  # Shape: (N1, N2, H)
        #print(f"e shape: {e.shape}")  # Debugging line to check shape
        return e

    def compute_attention_heads(self, Q, K, V, index_last_domain1=0, H1=0, H2=0):
        
        device = self.device
        dtype = self.dtype

        H, _, N = Q.shape
        # N, _, _ = V.shape  # Actually your code re-assigns same N but let's keep it as is
        _N, _, _ = V.shape
        index_first_domain2 = index_last_domain1 + 1

        # Get e from compute_product_Q_K
        e = self.compute_product_Q_K(Q, K)

        sf = torch.zeros(N, N, H, device=device, dtype=dtype)
        for h in range(H):
            if index_last_domain1 != 0:
                if h < H1:
                    # Heads for Domain 1
                    softmax_vals = torch.softmax(
                        e[0:index_last_domain1+1, 0:index_last_domain1+1, h], 
                        dim=1
                    )
                    top = torch.cat([
                        softmax_vals,
                        torch.zeros(
                            index_last_domain1+1, 
                            N - index_last_domain1 - 1, 
                            device=device
                        )
                    ], dim=1)
                    sf_domain = torch.cat([
                        top,
                        torch.zeros(
                            N - index_last_domain1 - 1, 
                            N, 
                            device=device
                        )
                    ], dim=0)
                    sf = sf.clone()
                    sf[:, :, h] = sf_domain
                elif h < H2:
                    # Heads for Domain 2
                    softmax_vals = torch.softmax(
                        e[index_first_domain2:, index_first_domain2:, h], 
                        dim=1
                    )
                    # Create the top-left zero block
                    bottom_left = torch.zeros(
                        N - index_first_domain2, 
                        index_first_domain2, 
                        device=device
                    )
                    # top and bottom
                    top = torch.zeros(index_first_domain2, N, device=device)
                    bottom = torch.cat([bottom_left, softmax_vals], dim=1)
                    sf_domain = torch.cat([top, bottom], dim=0)
                    sf = sf.clone()
                    sf[:, :, h] = sf_domain
                else:
                    # Heads for inter-domain interactions
                    sf_domain = torch.softmax(e[:, :, h], dim=1)
                    sf = sf.clone()
                    sf[:, :, h] = sf_domain
            else:
                # No domain masks applied
                sf_domain = torch.softmax(e[:, :, h], dim=1)
                sf = sf.clone()
                sf[:, :, h] = sf_domain

        return sf

    def compute_mat_ene(self, Q, K, V, Z, H1=0, H2=0, index_last_domain1=0):
        
        # We'll assume you intended to call self.compute_attention_heads here.
        # The old snippet references 'e' out of nowhere, so presumably that was from compute_product_Q_K.
        # We'll keep the lines exactly the same, except we clarify how 'sf' is obtained.

        # The code below uses variables that appear in the snippet,
        # but to keep it consistent, let's define them in place:
        device = self.device
        dtype = self.dtype

        H, q, _ = V.shape
        # For clarity in your snippet, 'e' was from compute_product_Q_K,
        # and 'sf' is from compute_attention_heads.
        # We'll compute them inside to match your logic:

        sf = self.compute_attention_heads(
            Q=Q,
            K=K,
            V=V,
            index_last_domain1=index_last_domain1,
            H1=H1,
            H2=H2
        )

        # From your snippet, you used e.shape for N_e1, N_e2, H_e,
        # but actually let's just read from sf itself.
        N_e1, N_e2, H_e = sf.shape
        N_Z, M = Z.shape

        assert N_e1 == N_e2 == N_Z, "Mismatch in N between sf and Z"
        N = N_e1

        # The code below uses variables that appear in the snippet,
        # but to keep it consistent, let's define them in place:
        device = self.device
        dtype = self.dtype

        H, q, _ = V.shape
        # For clarity in your snippet, 'e' was from compute_product_Q_K,
        # and 'sf' is from compute_attention_heads.
        # We'll compute them inside to match your logic:

        sf = self.compute_attention_heads(
            Q=Q, 
            K=K, 
            V=V, 
            index_last_domain1=index_last_domain1, 
            H1=H1, 
            H2=H2
        )

        # From your snippet, you used e.shape for N_e1, N_e2, H_e,
        # but actually let's just read from sf itself.
        N_e1, N_e2, H_e = sf.shape
        N_Z, M = Z.shape

        assert N_e1 == N_e2 == N_Z, "Mismatch in N between sf and Z"
        N = N_e1

        mat_ene = torch.zeros(N, q, M, device=device, dtype=dtype)

        # Weighted sum loop
        for h in range(H):
            V_h = V[h]
            # The next line in your snippet references V_h[:, Z], 
            # but that can be tricky because Z is shape (N, M).
            # We keep it as it is in your snippet, trusting you have reason:
            V_h_Zj = V_h[:, Z]     # shape => (q, N, M)
            V_h_Zj = V_h_Zj.permute(1, 0, 2)  # => (N, q, M)

            mat_ene_h = torch.einsum('ij,jqm->iqm', sf[:, :, h], V_h_Zj)
            mat_ene += mat_ene_h

        mat_ene = mat_ene.permute(1, 0, 2)
        return mat_ene, sf

    def compute_mat_ene_cond(self, Q, K, V, G, Z, P,g_scale, H1=0, H2=0, index_last_domain1=0):
        # Original attention/Potts energy
        mat_ene, sf = self.compute_mat_ene(Q, K, V, Z, H1=H1, H2=H2, index_last_domain1=index_last_domain1)
        # Conditioning term
        G_perm = self.G.permute(0, 2, 1)  # [N, f, q]
        bias = torch.einsum('bf,nfq->qnb', P, G_perm) 
        #bias = torch.einsum('bf,nfq->qnb', P, self.G)  # -> (q, N, batch_size)
        # Add conditioning bias
        mat_ene += g_scale * bias

        return mat_ene, sf

    def loss_cond(self, Q, K, V, G, Z, weights, P, g_scale, lambd=0.001, index_last_domain1=0, H1=0, H2=0):
        device = self.device
        dtype = self.dtype

        H, d, N = Q.shape
        _,N_plus_Ncomp, _ = K.shape  # K shape: (H, d, N+Ncomp)
        q = V.shape[1]  # Number of amino acids
        M = Z.shape[1]

        # Step: compute mat_ene and sf
        mat_ene, sf = self.compute_mat_ene_cond(self.Q, self.K, self.V, self.G, Z, P, g_scale, H1=H1, H2=H2, index_last_domain1=index_last_domain1)
        # logsumexp
        lge = torch.logsumexp(mat_ene, dim=0)  # Shape: (N, M)

        Z_indices = Z.unsqueeze(0)  # Shape: (1, N, M)
        mat_ene_selected = torch.gather(mat_ene, dim=0, index=Z_indices).squeeze(0)  # (N, M)

        pl_elements = weights * (mat_ene_selected - lge) #weighted sum along M
        pl = -torch.sum(pl_elements) # sum along N

        # For the regularization term, your snippet references M_matrix, etc.
        # That part of your snippet uses `self_mask`, but it was never fully spelled out. 
        # We'll keep it exactly as in your snippet:

        # The snippet tries: M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, self_mask)
        # But 'self_mask' is not defined in this scope. If that was part of your code, 
        # you must define it. We'll keep the line as is (though it may error if `self_mask` is missing).
    # Compute regularization term
        i_indices = torch.arange(N, device=device).unsqueeze(1)
        j_indices = torch.arange(N, device=device).unsqueeze(0)
        self_mask = (i_indices != j_indices).float()
        M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, self_mask)  # Shape: (H, H)
        VV = V.view(H, -1)  # Shape: (H, q*q)
        VV_T = VV @ VV.T  # Shape: (H, H)
        sum_J_squared = torch.sum(M_matrix * VV_T)  # Scalar
        reg = lambd * sum_J_squared  # Scalar
        reg_G = lambd * torch.sum(self.G**2)
        loss_value = pl + reg + reg_G

        del sf, mat_ene, mat_ene_selected, M, VV_T
        torch.cuda.empty_cache()
    
        return loss_value


