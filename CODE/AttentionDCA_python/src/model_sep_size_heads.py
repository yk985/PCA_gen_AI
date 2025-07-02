import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDomainAttentionSubBlock(nn.Module):
    """
    A single-layer attention model that splits H heads among 3 groups:
      - Domain1 heads (indices [0..H1)) => length domain1_end+1
      - Domain2 heads (indices [H1..H2)) => length N_beta
      - Inter-domain heads (indices [H2..H)) => domain1->domain2 + domain2->domain1

    Now each sub-block has its own Q/K/V parameters:
      - Q1,K1,V1 for domain1
      - Q2,K2,V2 for domain2
      - Qint1,Kint1,Vint1 for domain1->domain2
      - Qint2,Kint2,Vint2 for domain2->domain1
    """

    def __init__(
        self,
        H=32,          # total number of heads
        d=23,          # dimension for Q,K
        N=176,         # total protein length
        q=22,          # amino acid alphabet
        lambd=0.001,   # L2 reg
        domain1_end=63,# inclusive index for domain1
        H1=10,         # heads allocated for domain1
        H2=20,          # heads allocated for domain2, so interdomain = H-H2
        device = 'cpu',
        other_info_mat_ene = False
    ):
        super().__init__()
        self.H = H
        self.d = d
        self.N = N
        self.q = q
        self.lambd = lambd
        
        # domain1 is [0..domain1_end], length = (domain1_end+1)
        self.domain1_end = domain1_end
        self.domain2_start = domain1_end + 1
        self.N_alpha = self.domain2_start           # e.g. 64 if domain1_end=63
        self.N_beta  = self.N - self.N_alpha       
        self.device = device

        # heads indexing
        self.H1 = H1     # #heads for domain1
        self.H2 = H2     # #heads for domain1 + domain2 => domain2 is [H1..H2)
        self.other_info_mat_ene = other_info_mat_ene
        self.std_model = None
        # => interdomain heads are [H2..H)        

        # K = read_tensor_from_txt( cwd +"/results/34_23_save_random_without_J_400/K_tensor.txt")
        # Q= read_tensor_from_txt( cwd +"/results/34_23_save_random_without_J_400/Q_tensor.txt")
        # V = read_tensor_from_txt( cwd +"/results/34_23_save_random_without_J_400/V_tensor.txt")
        # # 1) Domain1 parameters: shape => (H1, d/q, L1)
        self.Q1 = nn.Parameter(torch.tensor(torch.randn(H1, d, self.N_alpha), device=self.device))
        self.K1 = nn.Parameter(torch.tensor(torch.randn(H1, d, self.N_alpha), device=self.device))
        self.V1 = nn.Parameter(torch.tensor(torch.randn(H1, q, q), device=self.device))

        # self.Q1.data = Q
        # self.K1.data = K
        # self.V1.data = V

        # 2) Domain2 parameters: shape => (H2-H1, d/q, L2)
        num_dom2_heads = self.H2 - self.H1
        self.Q2 = nn.Parameter(torch.randn(num_dom2_heads, d, self.N_beta))
        self.K2 = nn.Parameter(torch.randn(num_dom2_heads, d, self.N_beta))
        self.V2 = nn.Parameter(torch.randn(num_dom2_heads, q, q))

        # 3) Inter-domain parameters: shape => (H-H2, d/q, L1 or L2)
        num_inter_heads = self.H - self.H2
        num_inter_heads1 = num_inter_heads // 2
        num_inter_heads2 = num_inter_heads - num_inter_heads1
        # domain1->domain2
        self.Qint1 = nn.Parameter(torch.randn(num_inter_heads1, d, self.N_alpha))
        self.Kint1 = nn.Parameter(torch.randn(num_inter_heads1, d, self.N_beta))
        self.Vint1 = nn.Parameter(torch.randn(num_inter_heads1, q, q))
        # domain2->domain1
        self.Qint2 = nn.Parameter(torch.randn(num_inter_heads2, d, self.N_beta))
        self.Kint2 = nn.Parameter(torch.randn(num_inter_heads2, d, self.N_alpha))
        self.Vint2 = nn.Parameter(torch.randn(num_inter_heads2, q, q))

    def forward(self, Z, weights, head_group='domain1'):
        """
        Z: shape (N, M)
        weights: shape (M,) or (N, M)
        head_group in {'domain1','domain2','interdomain','all'}

        Returns:
          loss (scalar): negative pseudo-likelihood over the relevant sub-block(s).
        """
        device = Z.device
        N, M = Z.shape
        assert N == self.N, f"Mismatch: N={N}, self.N={self.N}"

        # For convenience, let's define which heads we process
        if head_group == 'domain1':
            head_indices = range(0, self.H1)
        elif head_group == 'domain2':
            head_indices = range(self.H1, self.H2)
        elif head_group == 'interdomain':
            head_indices = range(self.H2, self.H)
        else:  # 'all'
            head_indices = range(0, self.H)

        loss = torch.tensor(0.0, device=device)

        # If 'all', sum partial losses
        if head_group == 'domain1' or head_group == 'all':
            heads_d1 = range(0, self.H1) if head_group == 'all' else head_indices
            if len(heads_d1) > 0:
                sub_loss_d1 = self.domain1_attention(Z, weights, heads_d1)
                loss += sub_loss_d1
        
        if head_group == 'domain2' or head_group == 'all':
            heads_d2 = range(self.H1, self.H2) if head_group == 'all' else head_indices
            if len(heads_d2) > 0:
                sub_loss_d2 = self.domain2_attention(Z, weights, heads_d2)
                loss += sub_loss_d2
        
        if head_group == 'interdomain' or head_group == 'all':
            heads_int = range(self.H2, self.H) if head_group == 'all' else head_indices
            if len(heads_int) > 0:
                sub_loss_int = self.interdomain_attention(Z, weights, heads_int)
                loss += sub_loss_int

        return loss

    ###########################################################################
    #                          Domain1 sub-block
    ###########################################################################
    def construct_trained_domain1_model(self):



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
    
        
        H_old=120
        d_old=23
        import os
        cwd = os.getcwd()
        family_old = 'HKRR_30_10_80_JUSTdomain1_withHKRRtrainingfasta_d23_500batch'
        n_epochs_old=1000
        loss_type_old="without_J"
        # from domain1to1 already trained
        Q_old = read_tensor_from_txt( f"{cwd}/results/{H_old}_{d_old}_{family_old}_{loss_type_old}_{n_epochs_old}/Q1_tensor.txt" )
        K_old = read_tensor_from_txt( f"{cwd}/results/{H_old}_{d_old}_{family_old}_{loss_type_old}_{n_epochs_old}/K1_tensor.txt" )
        V_old = read_tensor_from_txt( f"{cwd}/results/{H_old}_{d_old}_{family_old}_{loss_type_old}_{n_epochs_old}/V1_tensor.txt" )
        Q_old = Q_old.to(self.device)
        K_old = K_old.to(self.device)
        V_old = V_old.to(self.device)
        from model import AttentionModel
        self.std_model = AttentionModel(H = Q_old.shape[0], d = Q_old.shape[1], N = Q_old.shape[2], q = V_old.shape[1], index_last_domain1=0, H1=0 , device=self.device)
        self.std_model.Q.data = Q_old
        self.std_model.K.data = K_old
        self.std_model.V.data = V_old
        self.std_model.to(self.device)


    def domain1_attention(self, Z, weights, head_indices):
        """
        Negative pseudo-likelihood for heads dedicated to domain1 => positions [0..N_alpha-1].
        """
        device = Z.device
        M = Z.shape[1]
        L1 = self.N_alpha  # e.g. domain1 length

        # Convert global head indices to local domain1 indices
        # e.g., if head_indices=[0,1,2] => local domain1 heads = the same [0,1,2]
        # because Q1 has shape (H1, d, L1)
        local_idx = [h for h in head_indices]  # no offset for domain1

        Q_sel = self.Q1[local_idx, :, :]  # (#heads_sel, d, L1)
        K_sel = self.K1[local_idx, :, :]
        V_sel = self.V1[local_idx, :, :]

        # e_sel: (L1, L1, #heads_sel)
        e_sel = torch.einsum('hdi,hdj->ijh', Q_sel, K_sel)

        # Exclude diagonal (i=j)
        mask_value = -1e9
        idx = torch.arange(L1, device=device)
        self_mask = (idx.unsqueeze(1) != idx.unsqueeze(0)).float()
        e_sel = e_sel * self_mask.unsqueeze(-1) + (1 - self_mask.unsqueeze(-1)) * mask_value

        # Only domain1 positions in Z
        Z_d1 = Z[:L1, :]  # shape (L1, M)

        mat_ene, sf = self.compute_mat_ene_subblock(e_sel, V_sel, Z_d1)
        # mat_ene: shape (q, L1, M)

        # Pseudo-likelihood
        lge = torch.logsumexp(mat_ene, dim=0)  # (L1, M)
        Z_indices = Z_d1.unsqueeze(0)          # (1, L1, M)
        mat_ene_sel = torch.gather(mat_ene, 0, Z_indices).squeeze(0)  # (L1, M)

        pl_elements = weights * (mat_ene_sel - lge)
        sub_loss = -torch.sum(pl_elements)

        # L2 reg
         
        H_sel, _, N_sel = Q_sel.shape
        i_indices = torch.arange(N_sel, device=device).unsqueeze(1)
        j_indices = torch.arange(N_sel, device=device).unsqueeze(0)
        self_mask = (i_indices != j_indices).float()
        M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, self_mask)  # Shape: (H, H)
        VV = V_sel.view(H_sel, -1)  # Shape: (H, q*q)
        VV_T = VV @ VV.T  # Shape: (H, H)
        reg_term = torch.sum(M_matrix * VV_T)  # Scalar
        #LOOK AT THE NORMALIZATION
        sub_loss += self.lambd * reg_term
        
        return sub_loss

    ###########################################################################
    #                          Domain2 sub-block
    ###########################################################################
    def domain2_attention(self, Z, weights, head_indices):
        """
        Negative pseudo-likelihood for domain2 heads => positions [domain2_start..N-1].
        Exclude self-interaction on the diagonal.
        """
        device = Z.device
        M = Z.shape[1]
        L2 = self.N_beta

        # domain2 heads are stored in Q2,K2,V2 with shape (H2-H1, d, L2)/(H2-H1, q, q).
        # Convert global heads to local domain2 indices => subtract H1
        local_idx = [h - self.H1 for h in head_indices]  # shift to [0..(H2-H1))

        Q_sel = self.Q2[local_idx, :, :]  # (#heads_sel, d, L2)
        K_sel = self.K2[local_idx, :, :]
        V_sel = self.V2[local_idx, :, :]

        # e_sel: (L2, L2, #heads_sel)
        e_sel = torch.einsum('hdi,hdj->ijh', Q_sel, K_sel)

        mask_value = -1e9
        idx = torch.arange(L2, device=device)
        self_mask = (idx.unsqueeze(1) != idx.unsqueeze(0)).float()
        e_sel = e_sel * self_mask.unsqueeze(-1) + (1 - self_mask.unsqueeze(-1)) * mask_value

        Z_d2 = Z[self.domain2_start:, :]  # shape (L2, M)

        mat_ene, sf = self.compute_mat_ene_subblock(e_sel, V_sel, Z_d2)
        # mat_ene: shape (q, L2, M)

        lge = torch.logsumexp(mat_ene, dim=0)   # (L2, M)
        Z_indices = Z_d2.unsqueeze(0)           # (1, L2, M)
        mat_ene_sel = torch.gather(mat_ene, 0, Z_indices).squeeze(0)  # (L2, M)

        pl_elements = weights * (mat_ene_sel - lge)
        sub_loss = -torch.sum(pl_elements)

        # L2 reg
         
        H_sel, _, N_sel = Q_sel.shape
        i_indices = torch.arange(N_sel, device=device).unsqueeze(1)
        j_indices = torch.arange(N_sel, device=device).unsqueeze(0)
        self_mask = (i_indices != j_indices).float()
        M_matrix = torch.einsum('ijh,ijk,ij->hk', sf, sf, self_mask)  # Shape: (H, H)
        VV = V_sel.view(H_sel, -1)  # Shape: (H, q*q)
        VV_T = VV @ VV.T  # Shape: (H, H)
        reg_term = torch.sum(M_matrix * VV_T)  # Scalar
        #LOOK AT THE NORMALIZATION
        sub_loss += self.lambd * reg_term
        
        return sub_loss

    ###########################################################################
    #                         Inter-domain sub-block
    ###########################################################################
    def interdomain_attention(self, Z, weights, head_indices):
        """
        Symmetrical cross-domain:
          1) domain1->domain2
          2) domain2->domain1
        Then sum both losses.
        """
        

        sub_loss_12 = self.cross_forward_domain1_to_domain2(Z, weights, head_indices)
        sub_loss_21 =0
        #sub_loss_21 = self.cross_forward_domain2_to_domain1(Z, weights, head_indices)
        return sub_loss_12 + sub_loss_21

    def cross_forward_domain1_to_domain2(self, Z, weights, head_indices):
        if self.other_info_mat_ene:
            mat_ene_domain1_trained, _ = self.std_model.compute_mat_ene(self.std_model.Q.data, self.std_model.K.data, self.std_model.V.data, Z[:self.N_alpha,:], H1=0, H2=0, index_last_domain1=0)
            #set mat_ene_domain1_trained grad to false
            mat_ene_domain1_trained.requires_grad = False
        else: mat_ene_domain1_trained = torch.zeros(self.q, self.N_alpha, Z.shape[1], device=Z.device)

        """
        domain1->domain2 uses Qint1, Kint1, Vint1
        """
        device = Z.device
        M = Z.shape[1]
        L1 = self.N_alpha
        L2 = self.N_beta
        head_indices1 = head_indices[:len(head_indices) // 2]
        #head_indices2 = head_indices[len(head_indices) // 2:]
        # Convert global heads to local interdomain indices => subtract H2
        local_idx = [h - self.H2 for h in head_indices1]

        Q_sel = self.Qint1[local_idx, :, :]  # (#heads_sel, d, L1)
        K_sel = self.Kint1[local_idx, :, :]  # (#heads_sel, d, L2)
        V_sel = self.Vint1[local_idx, :, :]  # (#heads_sel, q, q)

        # e_sel => shape (L1, L2, #heads_sel)
        e_sel = torch.einsum('hdi,hdj->ijh', Q_sel, K_sel)
        # Row-wise softmax along dimension=1 => L2
        e_sel = self.do_softmax_cross(e_sel, dim=1)

        Z_d1 = Z[:L1, :]
        Z_d2 = Z[self.domain2_start:, :]
        
        mat_ene = torch.zeros(self.q, L1, M, device=device, dtype=e_sel.dtype)
        
        for h in range(e_sel.shape[2]):
            sf_head = e_sel[:, :, h]  # (L1, L2)
            V_h = V_sel[h]            # (q, q)

            # V_h_Zd2 => shape (q, L2, M) => reorder => (L2, q, M)
            V_h_Zd2 = V_h[:, Z_d2]           # (q, L2, M)
            V_h_Zd2 = V_h_Zd2.permute(1, 0, 2)  # (L2, q, M)
            
            for i_idx in range(L1):
                row_weights = sf_head[i_idx]      # shape (L2,)
                weighted_V = torch.einsum('j,jqm->qm', row_weights, V_h_Zd2)
                mat_ene[:, i_idx, :] += weighted_V

        #HERE add mat_ene qxL1XM from domain1to1 already trained
        
        lge = torch.logsumexp(mat_ene+ mat_ene_domain1_trained, dim=0)  # (L1, M)
        Z_indices = Z_d1.unsqueeze(0)         # (1, L1, M)
        mat_ene_sel = torch.gather(mat_ene+mat_ene_domain1_trained, 0, Z_indices).squeeze(0)  # (L1, M)
        
        pl_elements = weights * (mat_ene_sel - lge)
        sub_loss = -torch.sum(pl_elements)

        sf = e_sel  # just rename for clarity
 
        M_matrix = torch.einsum('ijh,ijg->hg', sf, sf)

        # 2) Flatten each V_sel[h] => shape (#heads_sel, q^2)
        # Then do dot-product => (H_sel, H_sel)
        H_sel = V_sel.shape[0]
        VV = V_sel.view(H_sel, -1)        # (#heads_sel, q*q)
        VV_T = VV @ VV.T                  # (#heads_sel, #heads_sel)

        # 3) reg_term = sum_{h,h'} M_matrix[h,h'] * VV_T[h,h']
        reg_term = torch.sum(M_matrix * VV_T)

        # Weighted by lambd
        sub_loss += self.lambd * reg_term

        return sub_loss

    def cross_forward_domain2_to_domain1(self, Z, weights, head_indices):
        """
        domain2->domain1 uses Qint2, Kint2, Vint2
        """
        device = Z.device
        M = Z.shape[1]
        L1 = self.N_alpha
        L2 = self.N_beta
        head_indices1 = head_indices[:len(head_indices) // 2]
        head_indices2 = head_indices[len(head_indices) // 2:]
        local_idx = [h - self.H2-len(head_indices1) for h in head_indices2]

        Q_sel = self.Qint2[local_idx, :, :]  # (#heads_sel, d, L2)
        K_sel = self.Kint2[local_idx, :, :]  # (#heads_sel, d, L1)
        V_sel = self.Vint2[local_idx, :, :]  # (#heads_sel, q, q)

        # e_sel2 => shape (L2, L1, #heads_sel)
        e_sel2 = torch.einsum('hdi,hdj->ijh', Q_sel, K_sel)
        # softmax along dim=1 => the L1 axis
        e_sel2 = self.do_softmax_cross(e_sel2, dim=1)

        Z_d1 = Z[:L1, :]
        Z_d2 = Z[self.domain2_start:, :]

        mat_ene2 = torch.zeros(self.q, L2, M, device=device, dtype=e_sel2.dtype)
        
        for h in range(e_sel2.shape[2]):
            sf_head = e_sel2[:, :, h]  # (L2, L1)
            V_h = V_sel[h]            # (q, q)
            # V_h_Zd1 => shape (q, L1, M) => reorder => (L1, q, M)
            V_h_Zd1 = V_h[:, Z_d1].permute(1, 0, 2)  # (L1, q, M)
            
            for j_idx in range(L2):
                row_weights = sf_head[j_idx]  # (L1,)
                weighted_V = torch.einsum('i,iqm->qm', row_weights, V_h_Zd1)
                mat_ene2[:, j_idx, :] += weighted_V
        
        lge2 = torch.logsumexp(mat_ene2, dim=0)  # (L2, M)
        Z_indices2 = Z_d2.unsqueeze(0)           # (1, L2, M)
        mat_ene_sel2 = torch.gather(mat_ene2, 0, Z_indices2).squeeze(0)  # (L2, M)
        
        pl_elements2 = weights * (mat_ene_sel2 - lge2)
        sub_loss2 = -torch.sum(pl_elements2)

        sf = e_sel2  # just rename for clarity
 
        M_matrix = torch.einsum('ijh,ijg->hg', sf, sf)

        # 2) Flatten each V_sel[h] => shape (#heads_sel, q^2)
        # Then do dot-product => (H_sel, H_sel)
        H_sel = V_sel.shape[0]
        VV = V_sel.view(H_sel, -1)        # (#heads_sel, q*q)
        VV_T = VV @ VV.T                  # (#heads_sel, #heads_sel)

        # 3) reg_term = sum_{h,h'} M_matrix[h,h'] * VV_T[h,h']
        reg_term = torch.sum(M_matrix * VV_T)

        # Weighted by lambd
        sub_loss2 += self.lambd * reg_term

        return sub_loss2

    ###########################################################################
    #         Utility to build mat_ene for domain1 or domain2 sub-block
    ###########################################################################
    def compute_mat_ene_subblock(self, e_sel, V_sel, Z_sub):
        """
        e_sel: (L, L, #heads_sel)
        V_sel: (#heads_sel, q, q)
        Z_sub: shape (L, M)
        Returns:
          mat_ene: shape (q, L, M)
          sf:      shape (L, L, #heads_sel) [for debugging]
        """
        device = e_sel.device
        dtype = e_sel.dtype
        L, L2, num_heads = e_sel.shape
        _, q, q2 = V_sel.shape
        assert L == L2, "Square sub-block for domain attention"
        assert q == q2, "V must be (#heads_sel, q, q)"

        M = Z_sub.shape[1]
        
        mat_ene = torch.zeros(L, q, M, device=device, dtype=dtype)
        sf = torch.zeros(L, L, num_heads, device=device, dtype=dtype)

        for h in range(num_heads):
            sf_h = F.softmax(e_sel[:, :, h], dim=1)  # (L, L)
            sf[:, :, h] = sf_h
            V_h = V_sel[h]  # (q, q)
            
            V_h_Zj = V_h[:, Z_sub]         # shape (q, L, M)
            V_h_Zj = V_h_Zj.permute(1, 0, 2)  # (L, q, M)
            
            mat_ene_h = torch.einsum('ij,jqm->iqm', sf_h, V_h_Zj)  # => (L, q, M)
            mat_ene += mat_ene_h
        
        mat_ene = mat_ene.permute(1, 0, 2)  # (q, L, M)
        return mat_ene, sf

    def do_softmax_cross(self, e_sel, dim=1):
        """
        e_sel: shape (L1, L2, #heads) or (L2, L1, #heads)
        We do a softmax over the 'dim' dimension, per-head.
        """
        out = torch.zeros_like(e_sel)
        H_ = e_sel.shape[2]
        for h_idx in range(H_):
            out[..., h_idx] = F.softmax(e_sel[..., h_idx], dim=dim)
        return out