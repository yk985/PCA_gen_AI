import torch
import torch.nn as nn
import numpy as np

class AttentionModel_PCA(nn.Module):
    def __init__(
        self, 
        H, 
        d, 
        N1,#length of protein sequence
        N2, #number of PCA components
        q1, #number of aa
        q2, #number of PCA discretization 
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
        super(AttentionModel_PCA, self).__init__()

        # Device & dtype
        # You could choose your own logic for picking device, or force CPU/GPU.
        # For example:
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")  
        self.dtype = torch.float32  # or torch.float64, etc.

        # Store hyperparameters
        self.H = H
        self.d = d
        self.N1 = N1
        self.q1 = q1
        self.N2 = N2
        self.q2 = q2
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
            print("not working")
            self.Q = nn.Parameter(
                torch.tensor(np.random.randn(H, d, N1), dtype=self.dtype, device=self.device)
            )
        if K==None:
            self.K = nn.Parameter(
                torch.tensor(np.random.randn(H, d, N2), dtype=self.dtype, device=self.device)
            )
        if V==None:
            self.V = nn.Parameter(
                torch.tensor(np.random.randn(H, q1, q2), dtype=self.dtype, device=self.device)
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

        # Commented because we don't use masks for now 
        # # Exclude self-interactions by setting scores to -inf on the diagonal
        # i_indices = torch.arange(N, device=device).unsqueeze(1)
        # j_indices = torch.arange(N, device=device).unsqueeze(0)
        # self_mask = (i_indices != j_indices).float()
        # mask_value = -1e9  # A large negative value to zero out after softmax
        # e = e * self_mask.unsqueeze(-1) + (1 - self_mask.unsqueeze(-1)) * mask_value

        # # If there's a domain split:
        # if self.index_last_domain1 > 0 and self.index_last_domain1 < N:
        #     domain_masks = self.create_attention_masks(
        #         H=H, 
        #         L=N, 
        #         index_last_domain1=self.index_last_domain1,
        #         H1=self.H1,  # per your original usage
        #         H2=self.H2
        #     )
        #     # Invert the domain masks to identify positions to mask
        #     inverted_domain_masks = (1 - domain_masks).bool()  # Positions to mask are True
        #     # Permute e to match the shape of domain_masks
        #     e = e.permute(2, 0, 1)  # Shape: (H, N, N)
        #     # Apply the masks
        #     e = e.masked_fill(inverted_domain_masks, mask_value)
        #     # Permute e back to original shape
        #     e = e.permute(1, 2, 0)  # Shape: (N, N, H)
        # else:
        #     domain_masks = 0

        return e

    def compute_attention_heads(self, Q, K, V, index_last_domain1=0, H1=0, H2=0):
        
        device = self.device
        dtype = self.dtype
        #Not necessary
        # H, _, N = Q.shape
        # # N, _, _ = V.shape  # Actually your code re-assigns same N but let's keep it as is
        # _N, _, _ = V.shape
        index_first_domain2 = index_last_domain1 + 1

        # Get e from compute_product_Q_K
        e = self.compute_product_Q_K(Q, K)
        N1,N2,H=e.shape

        sf = torch.zeros(N1, N2, H, device=device, dtype=dtype)
        for h in range(H):
            if index_last_domain1 != 0:
                pass #Again no masks 
                # if h < H1:
                #     # Heads for Domain 1
                #     softmax_vals = torch.softmax(
                #         e[0:index_last_domain1+1, 0:index_last_domain1+1, h], 
                #         dim=1
                #     )
                #     top = torch.cat([
                #         softmax_vals,
                #         torch.zeros(
                #             index_last_domain1+1, 
                #             N - index_last_domain1 - 1, 
                #             device=device
                #         )
                #     ], dim=1)
                #     sf_domain = torch.cat([
                #         top,
                #         torch.zeros(
                #             N - index_last_domain1 - 1, 
                #             N, 
                #             device=device
                #         )
                #     ], dim=0)
                #     sf = sf.clone()
                #     sf[:, :, h] = sf_domain
                # elif h < H2:
                #     # Heads for Domain 2
                #     softmax_vals = torch.softmax(
                #         e[index_first_domain2:, index_first_domain2:, h], 
                #         dim=1
                #     )
                #     # Create the top-left zero block
                #     bottom_left = torch.zeros(
                #         N - index_first_domain2, 
                #         index_first_domain2, 
                #         device=device
                #     )
                #     # top and bottom
                #     top = torch.zeros(index_first_domain2, N, device=device)
                #     bottom = torch.cat([bottom_left, softmax_vals], dim=1)
                #     sf_domain = torch.cat([top, bottom], dim=0)
                #     sf = sf.clone()
                #     sf[:, :, h] = sf_domain
                # else:
                #     # Heads for inter-domain interactions
                #     sf_domain = torch.softmax(e[:, :, h], dim=1)
                #     sf = sf.clone()
                #     sf[:, :, h] = sf_domain
            else:
                # No domain masks applied
                sf_domain = torch.softmax(e[:, :, h], dim=1)
                sf = sf.clone()
                sf[:, :, h] = sf_domain

        return sf #shape (N1,N2,H)

    def compute_mat_ene(self, Q, K, V, Z, H1=0, H2=0, index_last_domain1=0):#A revoiiir: ajouter un autre Z ou juste separer dans la fonction pour protein sequence and PCA components
        
        # We'll assume you intended to call self.compute_attention_heads here.
        # The old snippet references 'e' out of nowhere, so presumably that was from compute_product_Q_K.
        # We'll keep the lines exactly the same, except we clarify how 'sf' is obtained.

        # The code below uses variables that appear in the snippet,
        # but to keep it consistent, let's define them in place:
        device = self.device
        dtype = self.dtype

        H, q1, q2 = V.shape
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
        q1,q2=V.shape
        # assert N_e1 == N_e2 == N_Z, "Mismatch in N between sf and Z"
        # N = N_e1

        mat_ene = torch.zeros(N_e1, q1, M, device=device, dtype=dtype)#q1 pas sur encore 

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

    def loss_wo_J(self, Q, K, V, Z, weights, lambd=0.001, index_last_domain1=0, H1=0, H2=0):
        
        device = self.device
        dtype = self.dtype

        H, d, N = Q.shape
        q = V.shape[1]  # Number of amino acids
        M = Z.shape[1]

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
        # i_indices = torch.arange(N, device=device).unsqueeze(1)
        # j_indices = torch.arange(N, device=device).unsqueeze(0)
        # self_mask = (i_indices != j_indices).float()
        #M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, self_mask)  # Shape: (H, H)
        M_matrix = torch.einsum('ijh,ijk->hk', sf, sf)  # Shape: (H, H)
        VV = V.view(H, -1)  # Shape: (H, q*q)
        VV_T = VV @ VV.T  # Shape: (H, H)
        sum_J_squared = torch.sum(M_matrix * VV_T)  # Scalar
        reg = lambd * sum_J_squared  # Scalar

        loss_value = pl + reg

        del sf, mat_ene, mat_ene_selected, M, VV_T
        torch.cuda.empty_cache()
    
        return loss_value



