import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# from model import loss_wo_J
from utils import quickread, get_sequences_pca_coords, get_PCA_grid_coords, one_hot_grid, add_PCA_coords
from dcascore import score, compute_PPV
from model import AttentionModel
from model_PCA_cond import ModelPCACond, ModelPCAcondJ
from sklearn.model_selection import train_test_split

class EarlyStopping:
    def __init__(self, patience=5, delta=0.05):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            #self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            #self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


def trainer_PCAcondJ(n_epochs, H=32, d=23, batch_size=1000, Nbins=35, Ncomp=2, eta=0.005, lambd=0.001,
            init_m=None, init_fun=np.random.randn,filename = None, structfile=None, verbose=True, savefile=None, losstype = 'without_J', index_last_domain1=0, H1=0, H2 =0,max_gap_frac=0.9):   

    Z, W = quickread(filename,max_gap_frac=max_gap_frac)
    print(f"Z shape: {Z.shape}, W shape: {W.shape}")
    Z = add_PCA_coords(Z.T,Nbins).T
    print(f"Z with PCA coords shape: {Z.shape}")
    W = W / W.sum()  # Normalize weights
    q = int(Z.max()) + 1  # Assuming Z contains 0-based indices
    N_plus_Ncomp, M = Z.shape
    N = N_plus_Ncomp - Ncomp
    print(N)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(f"Using device: {device}") 

    Z_train, Z_test, w_train, w_test = train_test_split(
        Z.T, W, test_size=0.2, random_state=42, shuffle=True
    )

    Z_train = torch.from_numpy(Z_train).long()
    Z_test = torch.from_numpy(Z_test).long()
    w_train = torch.from_numpy(w_train).float()
    w_test = torch.from_numpy(w_test).float()

    Z_train = Z_train.to(device)
    Z_test = Z_test.to(device)
    w_train = w_train.to(device)
    w_test = w_test.to(device)

    num_workers = 4 if device.type == 'cuda' else 0  # Adjust if necessary
    train_dataset = TensorDataset(Z_train, w_train)
    test_dataset = TensorDataset(Z_test, w_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    early_stopping = EarlyStopping(patience=10, delta=0.001)
    
    model = ModelPCAcondJ(         
        H, 
        d, 
        N, 
        q,
        Nbins=Nbins,
        Ncomp=Ncomp,
        lambd=0.001, 
        index_last_domain1=index_last_domain1, 
        H1=H1, 
        H2=H2, 
        init_fun=np.random.randn,
        device = device
        )
    if init_m is not None:
        model.Q.data = init_m.Q.data
        model.K.data = init_m.K.data
        model.V.data = init_m.V.data
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    def validate(epoch, train_losses=[]):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_z, batch_w in test_loader:
                batch_z = batch_z.to(device)
                batch_w = batch_w.to(device)
                loss = model(batch_z.T, batch_w)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    if savefile is not None:
        file = open(savefile, 'a')

    # Training Loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch_z, batch_w in train_loader:
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            batch_w = batch_w / batch_w.sum()
            optimizer.zero_grad()
            loss = model(batch_z.T, batch_w)
            loss = loss.mean()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_val_loss = validate(epoch, train_losses=train_losses)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Final Evaluation
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch_z, batch_w in test_loader:
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            loss = model(batch_z.T, batch_w)
            test_losses.append(loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return model

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_g_scale(epoch, total_epochs, g_scale_max):
    """
    Linearly ramp up g_scale from 0 to g_scale_max over the training epochs.
    """
    return g_scale_max * min(1.0, epoch / total_epochs)

import math

def get_g_scale_exp_ramp(epoch, total_epochs, g_scale_max=1.0):
    """
    Exponential ramp-up for G contribution.
    
    Args:
        epoch (int): current epoch (0-based)
        total_epochs (int): total number of training epochs
        g_scale_max (float): maximum scale for G
    
    Returns:
        float: scaled g_scale value
    """
    if epoch >= total_epochs:
        return g_scale_max
    phase = epoch / total_epochs
    return g_scale_max * math.exp(-5 * (1 - phase) ** 2)

def plot_pca_with_grid_and_counts(P, N, highlight_index=None):
    """
    Plot PCA coords P (Mx2) *without normalization*:
    - points colored by number of sequences in their grid cell
    - NxN grid lines overlayed based on PCA range
    - optional highlighted point by index
    """
    coords_min = P.min(axis=0)
    coords_max = P.max(axis=0)

    # Define bin edges for discretization
    x_bins = np.linspace(coords_min[0], coords_max[0], N+1)
    y_bins = np.linspace(coords_min[1], coords_max[1], N+1)

    # Digitize points to find which bin they fall into
    x_idx = np.digitize(P[:, 0], x_bins) - 1  # zero-based indices
    y_idx = np.digitize(P[:, 1], y_bins) - 1

    # Clip indices to ensure valid range [0, N-1]
    x_idx = np.clip(x_idx, 0, N-1)
    y_idx = np.clip(y_idx, 0, N-1)

    # Count number of points in each grid cell
    counts = np.zeros((N, N), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        counts[xi, yi] += 1

    # Map each point to its cell count
    point_counts = counts[x_idx, y_idx]

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(P[:, 0], P[:, 1], c=point_counts, s=40, cmap='plasma', label='Sequences (counts per cell)')
    plt.colorbar(scatter, label='Number of sequences in cell')

    # Draw grid lines
    for xt in x_bins:
        plt.axvline(xt, color='lightgray', linewidth=0.5)
    for yt in y_bins:
        plt.axhline(yt, color='lightgray', linewidth=0.5)

    # Highlight a specific point
    if highlight_index is not None:
        x, y = P[highlight_index]
        plt.scatter(x, y, color='red', s=100, label=f'Sequence {highlight_index}')
        plt.text(x + 0.01*(coords_max[0]-coords_min[0]), y + 0.01*(coords_max[1]-coords_min[1]), 
                 f'Index: {highlight_index}', color='red', fontsize=10)

    plt.title(f"PCA Projection with {N}x{N} Grid and Counts per Cell (raw coords)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(False)
    plt.show()

def trainer_PCA_cond(n_epochs, H=32, d=23, Nbins=20, g_scale=1.0, batch_size=1000, eta=0.005, lambd=0.001,
            init_m=None, init_fun=np.random.randn,filename = None, structfile=None, verbose=True, savefile=None, losstype = 'without_J', index_last_domain1=0, H1=0, H2 =0,max_gap_frac=0.9):   

    Z, W = quickread(filename,max_gap_frac=max_gap_frac)
    W = W / W.sum()  # Normalize weights
    q = int(Z.max()) + 1  # Assuming Z contains 0-based indices
    N, M = Z.shape
    print(N)
    P = one_hot_grid(Z, Nbins)  # Get PCA grid coordinates
    print(P.shape)
    #plot_pca_with_grid_and_counts(P, N=35, highlight_index=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    #device = 'cpu'
    print(f"Using device: {device}") 

    Z_train, Z_test, w_train, w_test, P_train, P_test = train_test_split(
        Z.T, W, P, test_size=0.2, random_state=42, shuffle=True
    )

    Z_train = torch.from_numpy(Z_train).long()
    Z_test = torch.from_numpy(Z_test).long()
    w_train = torch.from_numpy(w_train).float()
    w_test = torch.from_numpy(w_test).float()
    P_train = torch.from_numpy(P_train).float()
    P_test = torch.from_numpy(P_test).float()

    Z_train = Z_train.to(device)
    Z_test = Z_test.to(device)
    w_train = w_train.to(device)
    w_test = w_test.to(device)
    P_train = P_train.to(device)
    P_test = P_test.to(device)
    
    num_workers = 4 if device.type == 'cuda' else 0  # Adjust if necessary
    train_dataset = TensorDataset(Z_train, w_train, P_train)
    test_dataset = TensorDataset(Z_test, w_test, P_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    early_stopping = EarlyStopping(patience=5, delta=0.01) # 10,0.001
    # Initialize the model
    model = ModelPCACond(         
        H, 
        d, 
        N, 
        q, 
        Nbins,
        lambd=0.001, 
        index_last_domain1=index_last_domain1, 
        H1=H1, 
        H2=H2, 
        init_fun=np.random.randn,
        device = device
        )
    if init_m is not None:
        model.Q.data = init_m.Q.data
        model.K.data = init_m.K.data
        model.V.data = init_m.V.data
        model.G.data = init_m.G.data  # Initialize G if provided
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    def validate(epoch, train_losses=[]):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_z, batch_w, batch_p in test_loader:
                batch_z = batch_z.to(device)
                batch_w = batch_w.to(device)
                batch_p = batch_p.to(device)
                loss = model(batch_z.T, batch_w, batch_p)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    if savefile is not None:
        file = open(savefile, 'a')
    # Training Loop
    for epoch in range(1, n_epochs + 1):
        g_scale = get_g_scale_exp_ramp(epoch, n_epochs, g_scale)
        model.train()
        train_losses = []
        for batch_z, batch_w, batch_p in train_loader:
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            batch_p = batch_p.to(device)
            batch_w = batch_w / batch_w.sum()
            optimizer.zero_grad()
            loss = model(batch_z.T, batch_w, batch_p, g_scale=g_scale)
            loss = loss.mean()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_val_loss = validate(epoch, train_losses=train_losses)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Final Evaluation
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch_z, batch_w, batch_p in test_loader:
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            batch_p = batch_p.to(device)
            loss = model(batch_z.T, batch_w, batch_p)
            test_losses.append(loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return model