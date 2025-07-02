#do a random torch tensor of dimension N,N,H


def create_attention_masks(H, L, domain1_end, domain2_start, H1 , H2, device):
    # Initialize masks tensor
    masks = torch.zeros(H, L, L, device=device)

    # Define head indices for each type
    H_total = H


    # Create position indices
    positions = torch.arange(L, device=device)

    # Create domain masks
    domain1_mask = (positions.unsqueeze(1) <= domain1_end) & (positions.unsqueeze(0) <= domain1_end)
    domain2_mask = (positions.unsqueeze(1) >= domain2_start) & (positions.unsqueeze(0) >= domain2_start)
    inter_domain_mask = ((positions.unsqueeze(1) <= domain1_end) & (positions.unsqueeze(0) >= domain2_start)) | \
                        ((positions.unsqueeze(1) >= domain2_start) & (positions.unsqueeze(0) <= domain1_end))

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

import torch
N = 6
H = 3
device = 'cpu'
x = -1e-9
torch.manual_seed(0)

e = torch.tensor(
    [
     [[x,1,2,x,x,x], [3,x,4,x,x,x], [5,6,x,x,x,x], [x,x,x,x,x,x], [x,x,x,x,x,x], [x,x,x,x,x,x]],
     [[x,x,x,x,x,x], [x,x,x,x,x,x], [x,x,x,x,x,x],[x,x,x,x,7,8], [x,x,x,9,x,10], [x,x,x,11,12,x]],
     [[x,x,x, 13,14,15], [x,x,x,16,17,18], [x,x,x,19,20,21],[13,14,15,x,x,x], [16,17,18,x,x,x], [19,20,21,x,x,x]]
    ]
)
#e = torch.randn(H, N, N)

e = e.permute(1, 2, 0)

double_domain_masks = 2
H1 = 1
H2 = H1+1
# Exclude self-interactions by setting scores to -inf on the diagonal
i_indices = torch.arange(N, device=device).unsqueeze(1)
j_indices = torch.arange(N, device=device).unsqueeze(0)
self_mask = (i_indices != j_indices).float()
mask_value = -1e9  # A large negative value to zero out after softmax
e = e * self_mask.unsqueeze(-1) + (1 - self_mask.unsqueeze(-1)) * mask_value
# Apply domain masks if specified
if double_domain_masks > 0 and double_domain_masks < N:
    domain_masks = create_attention_masks(H, N, double_domain_masks, double_domain_masks + 1, H1, H2, device)
    # Invert the domain masks to identify positions to mask
    inverted_domain_masks = (1 - domain_masks).bool()  # Positions to mask are True
    # Permute e to match the shape of domain_masks
    e = e.permute(2, 0, 1)  # Shape: (H, N, N)
    # Apply the masks
    e = e.masked_fill(inverted_domain_masks, mask_value)
    # Permute e back to original shape
    e = e.permute(1, 2, 0)  # Shape: (N, N, H)
domain1_end = double_domain_masks
domain2_start = domain1_end + 1


sf = torch.zeros(N, N, H, device=device)


for h in range(H):
    if domain1_end != 0:
        if h < H1:
            # Heads for Domain 1
            softmax_vals = torch.softmax(e[0:domain1_end+1, 0:domain1_end+1, h], dim=1)
            top = torch.cat([
                softmax_vals,
                torch.zeros(domain1_end+1, N - domain1_end-1, device=device)
            ], dim=1)
            sf_domain = torch.cat([
                top,
                torch.zeros(N - domain1_end-1, N, device=device)
            ], dim=0)
            sf = sf.clone()
            sf[:, :, h] = sf_domain
        elif h < H2:
            # Heads for Domain 2
            softmax_vals = torch.softmax(e[domain2_start:, domain2_start:, h], dim=1)
            # Create the top-left zero block
            
            bottom_left = torch.zeros(N - domain2_start, domain2_start, device=device)  # 111x65

            
            # Concatenate top and bottom parts
            top = torch.zeros(domain2_start, N, device=device)     # 65x176
            bottom = torch.cat([bottom_left, softmax_vals], dim=1)  # 111x176

            # Concatenate top and bottom to form the final sf_domain
            sf_domain = torch.cat([top, bottom], dim=0)        # 176x176

            # Assign sf_domain to the corresponding head in sf
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


