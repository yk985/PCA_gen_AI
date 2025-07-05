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

    def forward(self, Z1,Z2, weights):
        # Forward just calls the loss, as in your original code
        loss_value = self.loss_wo_J_cross(
            self.Q, 
            self.K, 
            self.V, 
            Z1,
            Z2, 
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

    def compute_mat_ene_cross(self, Q, K, V, Z1, Z2, H1=0, H2=0, index_last_domain1=0):
        """
        Q: Tensor (H, d, N1)
        K: Tensor (H, d, N2)
        V: Tensor (H, q1, q2)
        Z1: LongTensor (N1, M)
        Z2: LongTensor (N2, M)
        sf: attention scores (computed from Q, K): (N1, N2, H)
        
        Returns:
            mat_ene: Tensor (M, N1) — energy matrix per sample and token
            sf: Tensor (N1, N2, H) — attention weights
        """
        device = self.device
        dtype = self.dtype

        H, q1, q2 = V.shape
        N1, M = Z1.shape
        N2 = Z2.shape[0]
        print("N2: ",N2)

        sf = self.compute_attention_heads(
            Q=Q, K=K, V=V, H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )  # shape: (N1, N2, H)

        mat_ene = torch.zeros(M, N1, device=device, dtype=dtype)  # Final energy: (M, N1)

        for h in range(H):
            V_h = V[h]  # shape: (q1, q2)

            # Z1: (N1, M) → (N1, 1, M)
            Z1_exp = Z1[:, None, :].expand(N1, N2, M)  # (N1, N2, M)
            Z2_exp = Z2[None, :, :].expand(N1, N2, M)  # (N1, N2, M)

            # Flatten to (N1*N2*M,)
            Z1_flat = Z1_exp.reshape(-1)
            Z2_flat = Z2_exp.reshape(-1)

            # Index into V_h[q1, q2] → result shape (N1*N2*M,)
            V_selected_flat = V_h[Z1_flat, Z2_flat]

            # Reshape back to (N1, N2, M)
            V_selected = V_selected_flat.view(N1, N2, M)

            sf_h = sf[:, :, h]  # (N1, N2)

            # Multiply sf_h * V_selected and sum over j (dim=1)
            mat_ene_h = torch.einsum('ij,ijm->mi', sf_h, V_selected)  # (M, N1)

            mat_ene += mat_ene_h

        return mat_ene, sf  # mat_ene: (M, N1)

    def loss_wo_J_cross(self, Q, K, V, Z1, Z2, weights, lambd=0.001, index_last_domain1=0, H1=0, H2=0):
        """
        Inputs:
            Q, K: (H, d, N1/N2)
            V: (H, q1, q2)
            Z1: (N1, M)
            Z2: (N2, M)
            weights: (M,)
        
        Returns:
            loss_value: scalar
        """
        device = self.device
        dtype = self.dtype

        H, d, N1 = Q.shape
        N2 = K.shape[2]
        q1, q2 = V.shape[1], V.shape[2]
        M = Z1.shape[1]

        mat_ene, sf = self.compute_mat_ene_cross(
            Q, K, V, Z1, Z2,
            H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )  # mat_ene: (M, N1)

        # 1. Gather selected true values: J_ij(Z1[i,m], Z2[j,m])
        # mat_ene[i,m] already contains ∑_j J_ij(Z1[i,m], Z2[j,m])

        # 2. Compute log-sum-exp normalization:
        # For each i, m: compute ∑_j J_ij(a, b_j^m) for all a ∈ [0, q1)
        # We'll loop over a ∈ [0, q1) and build (q1, N1, M)

        logZ = torch.zeros(N1, M, device=device, dtype=dtype)  # log partition

        for a in range(q1):
            log_terms = torch.zeros(N1, M, device=device, dtype=dtype)  # accumulator for each a

            for h in range(H):
                V_h = V[h]  # (q1, q2)
                V_ah = V_h[a]  # (q2,)
                V_ah_bjm = V_ah[Z2]  # (N2, M)

                sf_h = sf[:, :, h]  # (N1, N2)

                term = torch.einsum('ij,jm->im', sf_h, V_ah_bjm)  # (N1, M)
                log_terms += term

            logZ[a] = torch.logsumexp(log_terms, dim=0)  # sum over a later

        logZ = torch.logsumexp(logZ, dim=0)  # final shape: (M,)

        # 3. Loss = −sum(weights * (mat_ene - logZ))
        loss_per_sample = weights * (torch.sum(mat_ene, dim=1) - logZ)  # (M,)
        pl = -torch.sum(loss_per_sample)  # scalar

        # 4. Regularization (same as before)
        M_matrix = torch.einsum('ijh,ijk->hk', sf, sf)  # (H, H)
        VV = V.view(H, -1)  # (H, q1*q2)
        VV_T = VV @ VV.T  # (H, H)
        sum_J_squared = torch.sum(M_matrix * VV_T)  # scalar
        reg = lambd * sum_J_squared

        loss_value = pl + reg

        del sf, mat_ene, logZ, VV_T
        torch.cuda.empty_cache()

        return loss_value




