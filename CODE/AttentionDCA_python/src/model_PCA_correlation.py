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

    def forward(self, Q,K,V,Z1,Z2, weights):
        # Forward just calls the loss, as in your original code
        loss_value = self.compute_loss(
            Q,
            K,
            V,
            Z1,
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

        sf = torch.softmax(e, dim=1) 
        return sf #shape (N1,N2,H)

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
        i_indices = torch.arange(N).unsqueeze(1)
        j_indices = torch.arange(N).unsqueeze(0)
        self_mask = (i_indices != j_indices).float().to(Q.device)

        sf = sf * self_mask.unsqueeze(-1)
        # Gather all heads simultaneously
        V_Zj = V[:, :, Z] # (H, q, N, M)
        V_Zj = V_Zj.permute(2, 1, 3, 0) # (N, q, M, H)


        # Contract across N and H in one shot
        mat_ene = torch.einsum('ijh,jqmh->iqm', sf, V_Zj) # (N, q, M)
        mat_ene = mat_ene.permute(1, 0, 2)
        return mat_ene, sf
    def compute_mat_ene_cross(self, Q, K, V, Z1, Z2, H1=0, H2=0, index_last_domain1=0):
        """
        Q: Tensor (H, d, N1)
        K: Tensor (H, d, N2)
        V: Tensor (H, q1, q2)
        Z1: LongTensor (N1, M)
        Z2: LongTensor (N2, M)
        sf: attention scores (computed from Q, K): (N1, N2, H)
        
        Returns:
            mat_ene: Tensor (q1, N1, M) — same structure as compute_mat_ene
            sf: Tensor (N1, N2, H) — attention weights
        """
        device = self.device
        dtype = self.dtype

        H, q1, q2 = V.shape
        N1, M = Z1.shape
        N2 = Z2.shape[0]

        sf = self.compute_attention_heads(
            Q=Q, K=K, V=V, H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )  # shape: (N1, N2, H)

        # Final energy: (q1, N1, M)  (same as compute_mat_ene)
        mat_ene = torch.zeros(q1, N1, M, device=device, dtype=dtype)

        # ---- Gather V across q2 dimension using Z2 ----
        Z2_exp = Z2.unsqueeze(0).unsqueeze(0)               # (1, 1, N2, M)
        Z2_exp = Z2_exp.expand(H, q1, N2, M)                # (H, q1, N2, M)

        V_exp = V.unsqueeze(2).unsqueeze(3).expand(H, q1, N2, M, q2)  # (H, q1, N2, M, q2)
        V_Z2 = torch.gather(V_exp, -1, Z2_exp.unsqueeze(-1))          # (H, q1, N2, M, 1)
        V_Z2 = V_Z2.squeeze(-1)                                       # (H, q1, N2, M)

        V_Z2 = V_Z2.permute(1, 2, 3, 0)   # (q1, N2, M, H)

        # Expand across N1 to align with sf
        V_Z2 = V_Z2.unsqueeze(1).expand(q1, N1, N2, M, H)  # (q1, N1, N2, M, H)

        # ---- Weighted sum ----
        mat_ene = torch.einsum('nkh,qnkmh->qnm', sf, V_Z2)  # (q1, N1, M)

        return mat_ene, sf  # mat_ene: (q1, N1, M)

    def compute_loss(self, QJ, KJ, VJ, Z, 
                        QG, KG, VG, Z1, Z2, weights,lambd=0.001,
                        H1=0, H2=0, index_last_domain1=0):
        """
        Computes the loss L(J) from the given formula.

        Args:
            QJ, KJ, VJ: tensors for J computation (self energy)
            Z: LongTensor (N1, M), indices for J
            QG, KG, VG: tensors for G computation (cross energy)
            Z1: LongTensor (N1, M), indices for sequence 1
            Z2: LongTensor (N2, M), indices for sequence 2
            H1, H2, index_last_domain1: attention settings

        Returns:
            loss: scalar tensor
        """

        # ------------------------
        # Step 1: Compute J (self-energy)
        # ------------------------
        # mat_ene_J: (q1, N1, M)+regularization
        mat_ene_J, sf_J = self.compute_mat_ene(
            Q=QJ, K=KJ, V=VJ, Z=Z,
            H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )
        
        H, d, N1 = QJ.shape
        M_matrix = torch.einsum('ijh,ijk->hk', sf_J, sf_J)  # (H, H)
        VV = VJ.view(H, -1)  # (H, q1*q2)
        VV_T = VV @ VV.T  # (H, H)
        sum_J_squared = torch.sum(M_matrix * VV_T)  # scalar
        reg1 = lambd * sum_J_squared
        # ------------------------
        # Step 2: Compute G (cross-energy)
        # ------------------------
        # mat_ene_G: (q1, N1, M)+regularization
        mat_ene_G, sf_G = self.compute_mat_ene_cross(
            Q=QG, K=KG, V=VG, Z1=Z1, Z2=Z2,
            H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )
        H, d, N1 = QG.shape
        M_matrix = torch.einsum('ijh,ijk->hk', sf_G, sf_G)  # (H, H)
        VV = VG.view(H, -1)  # (H, q1*q2)
        VV_T = VV @ VV.T  # (H, H)
        sum_G_squared = torch.sum(M_matrix * VV_T)  # scalar
        reg2 = lambd * sum_G_squared
        # ------------------------
        # Step 3: Positive terms
        # ------------------------
        # Pick correct index a_i^m from mat_ene_J
        # Z: (N1, M), we gather along dim=0
        # -> pos_J: (N1, M)
        q1, N1, M = mat_ene_J.shape
        Z_t = Z.T.unsqueeze(0)  # (1, M, N1)
        pos_J = mat_ene_J.permute(2,1,0).gather(
            dim=2, index=Z.T.unsqueeze(-1)
        ).squeeze(-1).permute(1,0)  # (N1, M)

        # Pick correct index a_i^m from mat_ene_G
        pos_G = mat_ene_G.permute(2,1,0).gather(
            dim=2, index=Z.T.unsqueeze(-1)
        ).squeeze(-1).permute(1,0)  # (N1, M)

        # Positive contribution: sum over i
        pos_term = pos_J + pos_G  # (N1, M)

        # ------------------------
        # Step 4: Log partition function
        # ------------------------
        # For each (i,m), compute:
        # logsumexp over a ∈ [1..q1]
        # exp( mat_ene_J[a,i,m] + mat_ene_G[a,i,m] )
        logits = mat_ene_J + mat_ene_G  # (q1, N1, M)
        logZ = torch.logsumexp(logits, dim=0)  # (N1, M)

        # ------------------------
        # Step 5: Final loss
        # ------------------------
        #loss_matrix =weights*(pos_term - logZ)  # (N1, M)
        loss_matrix= weights*torch.sum(pos_term-logZ,dim=0)
        loss = -loss_matrix.mean()  # scalar
        loss+=reg2+reg1
        del sf_J,sf_G, mat_ene_J,mat_ene_G, loss_matrix, VV_T, VV, M_matrix
        torch.cuda.empty_cache()

        
        return loss


class AttentionModel_PCA_once(nn.Module):
    def __init__(
        self, 
        H, 
        d1, 
        N1, 
        q1,
        H_PCA, 
        d2, 
        N2, 
        q2, 
        Q=None,
        V=None,
        K=None,
        Q_PCA=None,
        V_PCA=None,
        K_PCA=None,
        lambd=0.001, 
        index_last_domain1=0, 
        H1=0, 
        H2=0, 
        init_fun=np.random.randn,
        device = 'cpu'
        
    ):
        super(AttentionModel_PCA_once, self).__init__()

        # Device & dtype
        # You could choose your own logic for picking device, or force CPU/GPU.
        # For example:
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")  
        self.dtype = torch.float32  # or torch.float64, etc.

        # Store hyperparameters
        self.H = H
        self.d1 = d1
        self.N1 = N1
        self.q1 = q1
        self.H_PCA = H_PCA
        self.d2 = d2
        self.N2= N2
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
            self.Q = nn.Parameter(
                torch.tensor(np.random.randn(H, d1, N1), dtype=self.dtype, device=self.device)
            )
        if K==None:
            self.K = nn.Parameter(
                torch.tensor(np.random.randn(H, d1, N1), dtype=self.dtype, device=self.device)
            )
        if V==None:
            self.V = nn.Parameter(
                torch.tensor(np.random.randn(H, q1, q1), dtype=self.dtype, device=self.device)
            )
        if Q_PCA==None:
            self.Q_PCA = nn.Parameter(
                torch.tensor(np.random.randn(H_PCA, d2, N1), dtype=self.dtype, device=self.device)
            )
        if K_PCA==None:
            self.K_PCA = nn.Parameter(
                torch.tensor(np.random.randn(H_PCA, d2, N2), dtype=self.dtype, device=self.device)
            )
        if V_PCA==None:
            self.V_PCA = nn.Parameter(
                torch.tensor(np.random.randn(H_PCA, q1, q2), dtype=self.dtype, device=self.device)
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
    
        loss_value=self.compute_loss(self.Q,self.K,self.V,Z1,self.Q_PCA,self.K_PCA,self.V_PCA,Z1,Z2,weights,lambd=self.lambd,index_last_domain1=self.index_last_domain1,H1=self.H1,H2=self.H2)
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
        i_indices = torch.arange(N).unsqueeze(1)
        j_indices = torch.arange(N).unsqueeze(0)
        self_mask = (i_indices != j_indices).float().to(Q.device)

        sf = sf * self_mask.unsqueeze(-1)
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
    def compute_mat_ene_cross(self, Q, K, V, Z1, Z2, H1=0, H2=0, index_last_domain1=0):
        """
        Q: Tensor (H, d, N1)
        K: Tensor (H, d, N2)
        V: Tensor (H, q1, q2)
        Z1: LongTensor (N1, M)
        Z2: LongTensor (N2, M)
        sf: attention scores (computed from Q, K): (N1, N2, H)
        
        Returns:
            mat_ene: Tensor (q1, N1, M) — same structure as compute_mat_ene
            sf: Tensor (N1, N2, H) — attention weights
        """
        device = self.device
        dtype = self.dtype

        H, q1, q2 = V.shape
        N1, M = Z1.shape
        N2 = Z2.shape[0]

        sf = self.compute_attention_heads(
            Q=Q, K=K, V=V, H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )  # shape: (N1, N2, H)

        # Final energy: (q1, N1, M)  (same as compute_mat_ene)
        mat_ene = torch.zeros(q1, N1, M, device=device, dtype=dtype)

        for h in range(H):
            V_h = V[h]  # (q1, q2)

            # Z1: (N1, M) → (N1, 1, M)
            Z1_exp = Z1[:, None, :].expand(N1, N2, M)  # (N1, N2, M)
            Z2_exp = Z2[None, :, :].expand(N1, N2, M)  # (N1, N2, M)

            # Flatten to (N1*N2*M,)
            Z1_flat = Z1_exp.reshape(-1)
            Z2_flat = Z2_exp.reshape(-1)

            # Index into V_h[q1, q2] → (N1*N2*M, q1)
            V_selected_flat = V_h[:, Z2_flat]  # (q1, N1*N2*M)

            # Reshape to (q1, N1, N2, M)
            V_selected = V_selected_flat.view(q1, N1, N2, M)

            sf_h = sf[:, :, h]  # (N1, N2)

            # Weighted sum over j (dim=2, the N2 dimension)
            mat_ene_h = torch.einsum('ij,qijm->qim', sf_h, V_selected)  # (q1, N1, M)

            mat_ene += mat_ene_h

        return mat_ene, sf  # mat_ene: (q1, N1, M)


    
    def compute_loss(self, QJ, KJ, VJ, Z, 
                        QG, KG, VG, Z1, Z2, weights,lambd=0.001,
                        H1=0, H2=0, index_last_domain1=0):
        """
        Computes the loss L(J) from the given formula.

        Args:
            QJ, KJ, VJ: tensors for J computation (self energy)
            Z: LongTensor (N1, M), indices for J
            QG, KG, VG: tensors for G computation (cross energy)
            Z1: LongTensor (N1, M), indices for sequence 1
            Z2: LongTensor (N2, M), indices for sequence 2
            H1, H2, index_last_domain1: attention settings

        Returns:
            loss: scalar tensor
        """

        # ------------------------
        # Step 1: Compute J (self-energy)
        # ------------------------
        # mat_ene_J: (q1, N1, M)+regularization
        mat_ene_J, sf_J = self.compute_mat_ene(
            Q=QJ, K=KJ, V=VJ, Z=Z,
            H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )
        
        H, d, N1 = QJ.shape
        M_matrix = torch.einsum('ijh,ijk->hk', sf_J, sf_J)  # (H, H)
        VV = VJ.view(H, -1)  # (H, q1*q2)
        VV_T = VV @ VV.T  # (H, H)
        sum_J_squared = torch.sum(M_matrix * VV_T)  # scalar
        reg1 = lambd * sum_J_squared
        # ------------------------
        # Step 2: Compute G (cross-energy)
        # ------------------------
        # mat_ene_G: (q1, N1, M)+regularization
        mat_ene_G, sf_G = self.compute_mat_ene_cross(
            Q=QG, K=KG, V=VG, Z1=Z1, Z2=Z2,
            H1=H1, H2=H2, index_last_domain1=index_last_domain1
        )
        H, d, N1 = QG.shape
        M_matrix = torch.einsum('ijh,ijk->hk', sf_G, sf_G)  # (H, H)
        VV = VG.view(H, -1)  # (H, q1*q2)
        VV_T = VV @ VV.T  # (H, H)
        sum_G_squared = torch.sum(M_matrix * VV_T)  # scalar
        reg2 = lambd * sum_G_squared
        # ------------------------
        # Step 3: Positive terms
        # ------------------------
        # Pick correct index a_i^m from mat_ene_J
        # Z: (N1, M), we gather along dim=0
        # -> pos_J: (N1, M)
        q1, N1, M = mat_ene_J.shape
        Z_t = Z.T.unsqueeze(0)  # (1, M, N1)
        pos_J = mat_ene_J.permute(2,1,0).gather(
            dim=2, index=Z.T.unsqueeze(-1)
        ).squeeze(-1).permute(1,0)  # (N1, M)

        # Pick correct index a_i^m from mat_ene_G
        pos_G = mat_ene_G.permute(2,1,0).gather(
            dim=2, index=Z.T.unsqueeze(-1)
        ).squeeze(-1).permute(1,0)  # (N1, M)

        # Positive contribution: sum over i
        pos_term = pos_J + pos_G  # (N1, M)

        # ------------------------
        # Step 4: Log partition function
        # ------------------------
        # For each (i,m), compute:
        # logsumexp over a ∈ [1..q1]
        # exp( mat_ene_J[a,i,m] + mat_ene_G[a,i,m] )
        logits = mat_ene_J + mat_ene_G  # (q1, N1, M)
        logZ = torch.logsumexp(logits, dim=0)  # (N1, M)

        # ------------------------
        # Step 5: Final loss
        # ------------------------
        #loss_matrix =weights*(pos_term - logZ)  # (N1, M)
        loss_matrix= weights*torch.sum(pos_term-logZ,dim=0)
        loss = -loss_matrix.mean()  # scalar
        loss+=reg2+reg1
        del sf_J,sf_G, mat_ene_J,mat_ene_G, loss_matrix, VV_T
        torch.cuda.empty_cache()

        
        return loss

