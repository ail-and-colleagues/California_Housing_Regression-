import datetime
import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from dataset import Dataset_Template
from model import Reg_Model


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = datetime.datetime.now().strftime("%Y%m%d_%H_%M_")
    output_dir = os.path.join('./outputs', output_dir)

    os.makedirs(output_dir, exist_ok=True)  

    data_path = "./datasets/cadata.txt"
    batch_size = 1024
    batch_p_ep = 100
    latent_feature_dim = 5
    input_dim = 8
    output_dim = 1
    mx_ep = 100

    train_set = Dataset_Template(data_path, batch_size, batch_p_ep)
    # Tentatively, the val_set uses the same data as the train_set.
    # The val_set must be different from the train_set In the actual training.
    val_set = Dataset_Template(data_path, batch_size, 1)

    reg_model = Reg_Model(input_dim, latent_feature_dim, output_dim).to(device)
    summary(reg_model)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    criterion = MSELoss()
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=0.01)

    for ep in range(mx_ep):
        losses = list()

        with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train_loader:
            for i, (x, y) in tqdm_train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                pred, lf = reg_model(x)
                loss = criterion(pred, y)
                
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                loss_mean = np.mean(losses)
                tqdm_train_loader.set_description("[Epoch %d]" % (ep))
                tqdm_train_loader.set_postfix(OrderedDict(loss=loss_mean))

        with torch.no_grad():
            x, y = next(iter(val_loader))
            x = x.to(device)
            y = y.to(device)
            pred, lf = reg_model(x)
            loss = criterion(pred, y)
            loss = loss.item()
            print('val loss: ', loss)

            dst = os.path.join(output_dir, 'ep{:03d}_loss{:.3f}_{:.3f}.pth'.format(ep, loss_mean, loss))
            torch.save(reg_model.state_dict(), dst)

            

if __name__ == "__main__":
    train()