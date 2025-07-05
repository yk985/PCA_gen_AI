import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# from model import loss_wo_J
from utils import quickread, add_PCA_coords
from dcascore import score, compute_PPV
from model import AttentionModel
from model_PCA_correlation import AttentionModel_PCA
from sklearn.model_selection import train_test_split

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
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

def trainer(n_epochs, H=32, d=23, batch_size=1000, eta=0.005, lambd=0.001,
            init_m=None, init_fun=np.random.randn,filename = None, structfile=None, verbose=True, savefile=None, losstype = 'without_J', index_last_domain1=0, H1=0, H2 =0,max_gap_frac=0.9):   

    Z, W = quickread(filename,max_gap_frac=max_gap_frac)
    W = W / W.sum()  # Normalize weights
    q = int(Z.max()) + 1  # Assuming Z contains 0-based indices
    N, M = Z.shape
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
    
    model = AttentionModel(         
        H, 
        d, 
        N, 
        q, 
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


def trainer_PCA_comp_brute_force(n_epochs, H=32, d=23, batch_size=1000, eta=0.005, lambd=0.001,
            init_m=None, init_fun=np.random.randn,filename = None, structfile=None, verbose=True, savefile=None, losstype = 'without_J', index_last_domain1=0, H1=0, H2 =0,max_gap_frac=0.9,nb_bins_PCA=35):   

    Z, W = quickread(filename,max_gap_frac=max_gap_frac)
    Z=add_PCA_coords(Z.T,nb_bins_PCA).T
    W = W / W.sum()  # Normalize weights
    q = int(Z.max()) + 1  # Assuming Z contains 0-based indices
    N, M = Z.shape
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
    
    model = AttentionModel(         
        H, 
        d, 
        N, 
        q, 
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

def trainer_PCA_comp_2_model(n_epochs, H=32, d=23, batch_size=1000, eta=0.005, lambd=0.001,
            init_m=None, init_fun=np.random.randn,filename = None, structfile=None, verbose=True, savefile=None, losstype = 'without_J', index_last_domain1=0, H1=0, H2 =0,max_gap_frac=0.9,nb_bins_PCA=35):   

    Z, W = quickread(filename,max_gap_frac=max_gap_frac)
    Z=add_PCA_coords(Z.T,nb_bins_PCA).T
    Z1=Z[:-2,:]
    Z2=Z[-2:,:]
    W = W / W.sum()  # Normalize weights
    q1 = int(Z1.max()) + 1  # Assuming Z contains 0-based indices
    q2 = int(Z2.max()) + 1 # à checker avec marzio pour index 0 ou pas 
    N1, M = Z1.shape
    N2, _ = Z2.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(f"Using device: {device}") 

    Z1_train, Z1_test,Z2_train, Z2_test, w_train, w_test = train_test_split(
        Z1.T,Z2.T, W, test_size=0.2, random_state=42, shuffle=True
    )

    Z1_train = torch.from_numpy(Z1_train).long()
    Z1_test = torch.from_numpy(Z1_test).long()
    Z2_train = torch.from_numpy(Z2_train).long()
    Z2_test = torch.from_numpy(Z2_test).long()
    w_train = torch.from_numpy(w_train).float()
    w_test = torch.from_numpy(w_test).float()

    Z1_train = Z1_train.to(device)
    Z1_test = Z1_test.to(device)
    Z2_train = Z2_train.to(device)
    Z2_test = Z2_test.to(device)
    w_train = w_train.to(device)
    w_test = w_test.to(device)

    num_workers = 4 if device.type == 'cuda' else 0  # Adjust if necessary
    train_dataset = TensorDataset(Z1_train,Z2_train, w_train)
    test_dataset = TensorDataset(Z1_test, Z2_test ,w_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    early_stopping = EarlyStopping(patience=10, delta=0.001)
    
    model = AttentionModel_PCA(         
        H, 
        d, 
        N1,
        N2, 
        q1,
        q2, 
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
            for batch_z1, batch_z2 ,batch_w in test_loader:
                batch_z1 = batch_z1.to(device)
                batch_z2 = batch_z2.to(device)
                batch_w = batch_w.to(device)
                loss = model(batch_z1.T,batch_z2.T , batch_w)
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
        for batch_z1,batch_z2, batch_w in train_loader:
            batch_z1 = batch_z1.to(device)
            batch_z2 = batch_z2.to(device)
            batch_w = batch_w.to(device)
            batch_w = batch_w / batch_w.sum()
            optimizer.zero_grad()
            loss = model(batch_z1.T,batch_z2.T ,batch_w)
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
        for batch_z1, batch_z2, batch_w in test_loader:
            batch_z1 = batch_z1.to(device)
            batch_z2 = batch_z2.to(device)
            batch_w = batch_w.to(device)
            loss = model(batch_z1.T,batch_z2.T, batch_w)
            test_losses.append(loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return model