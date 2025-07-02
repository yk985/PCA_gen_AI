import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_sep_size_heads import MultiDomainAttentionSubBlock
import torch.nn as nn
from utils import quickread
from sklearn.model_selection import train_test_split

class EarlyStopping:
    """
    Example early stopping class, same as your code.
    """
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



def trainer_multidomain_strategyB(n_epochs=10,
                                 H=32, d=23,
                                 batch_size=1000,
                                 eta=0.005, lambd=0.001,
                                 domain1_end=64,      # inclusive => domain1 = [0..64]
                                 H1=10, H2=29,        # example partition among heads
                                 filename=None,
                                 other_info_mat_ene=False
                                 ):
    """
    1) Loads MSA data (Z, W).
    2) Splits into train/test.
    3) Creates a MultiDomainAttentionSubBlock with symmetrical inter-domain.
    4) Per mini-batch, do domain1->update, domain2->update, inter->update.
    5) EarlyStopping on validation.
    """
    # 1) Load data
    Z, W = quickread(filename)
    W = W / W.sum()
    q = int(Z.max()) + 1

    N, M = Z.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(f"Using device: {device}")

    # 2) train-test split
    Z_train, Z_test, w_train, w_test = train_test_split(
        Z.T, W, test_size=0.2, random_state=42, shuffle=True
    )
    Z_train = torch.from_numpy(Z_train).long()
    Z_test  = torch.from_numpy(Z_test).long()
    w_train = torch.from_numpy(w_train).float()
    w_test  = torch.from_numpy(w_test).float()

    train_dataset = TensorDataset(Z_train, w_train)
    test_dataset  = TensorDataset(Z_test, w_test)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 3) Create the model
    from model_sep_size_heads import MultiDomainAttentionSubBlock  # or the model code above
    model = MultiDomainAttentionSubBlock(
        H=H, d=d, N=N, q=q, lambd=lambd,
        domain1_end=domain1_end,
        H1=H1, H2=H2, device=device,
        other_info_mat_ene = other_info_mat_ene
    ).to(device)
    if other_info_mat_ene:
        model.construct_trained_domain1_model()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    optimizer = optimizer = torch.optim.Adam(
    [
    #    {'params': model.Q1, 'weight_decay': 0.0}, 
    #    {'params': model.K1, 'weight_decay': 0.0},
    #    {'params': model.V1, 'weight_decay': 0.0},
    #     {'params': model.Q2, 'weight_decay': 0.0}, 
    #    {'params': model.K2, 'weight_decay': 0.0},
    #    {'params': model.V2, 'weight_decay': 0.0},
       {'params': model.Qint1, 'weight_decay': 0.0}, 
       {'params': model.Kint1, 'weight_decay': 0.0},
       {'params': model.Vint1, 'weight_decay': 0.0}
    #     {'params': model.Qint2, 'weight_decay': 0.0}, 
    #    {'params': model.Kint2, 'weight_decay': 0.0},
    #    {'params': model.Vint2, 'weight_decay': 0.0}
       # no domain2, no interdomain
    ],
    lr=eta
)
    # model.Q2.requires_grad_(False)
    # model.K2.requires_grad_(False)
    # model.V2.requires_grad_(False)
    # model.Qint1.requires_grad_(False)
    # model.Kint1.requires_grad_(False)
    # model.Vint1.requires_grad_(False)
    # model.Qint2.requires_grad_(False)
    # model.Kint2.requires_grad_(False)
    # model.Vint2.requires_grad_(False)


    early_stopping = EarlyStopping(patience=10, delta=0.001)

    def validate(epoch, train_losses=None):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_z, batch_w in test_loader:
                batch_z = batch_z.to(device)
                batch_w = batch_w.to(device)
                loss_val = model(batch_z.T, batch_w, head_group='interdomain') #ALL normally
                val_losses.append(loss_val.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_loss = (sum(train_losses)/len(train_losses)) if train_losses else float('nan')
        print(f"[Epoch {epoch}] TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}")
        return avg_val_loss

    # 4) Training loop
    for epoch in range(1, n_epochs+1):
        model.train()
        train_losses = []
        
        for batch_z, batch_w in train_loader:
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            batch_w = batch_w / batch_w.sum()                

            #domain1 update
            # optimizer.zero_grad()
            # loss_d1 = model(batch_z.T, batch_w, head_group='domain1')
            # loss_d1.backward()
            # optimizer.step()

            # # domain2 update
            # optimizer.zero_grad()
            # loss_d2 = model(batch_z.T, batch_w, head_group='domain2')
            # loss_d2.backward()
            # optimizer.step()

            # inter-domain update (symmetrical cross)
            optimizer.zero_grad()
            loss_int = model(batch_z.T, batch_w, head_group='interdomain')
            loss_int.backward()
            optimizer.step()

            loss_d2 =0
            loss_d1 =0
            #loss_int = 0

            total_loss = (loss_d1 + loss_d2 + loss_int).item()
            train_losses.append(total_loss)

        avg_val_loss = validate(epoch, train_losses)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # final test
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch_z, batch_w in test_loader:
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            loss_test = model(batch_z.T, batch_w, head_group='interdomain')#+model(batch_z.T, batch_w, head_group='domain1') #ALL normally
            test_losses.append(loss_test.item())
    avg_test_loss = sum(test_losses)/len(test_losses)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    return model