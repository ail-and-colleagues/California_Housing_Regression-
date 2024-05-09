import torch
import torch.nn.functional as F

from torchinfo import summary

class Reg_Model(torch.nn.Module):
    def __init__(self, input_dim, latent_feature_dim, output_dim):
        super().__init__()
        # implemetation
        self.input_dim = input_dim
        self.latent_feature_dim = latent_feature_dim
        self.output_dim = output_dim

        self.linear1 = torch.nn.Linear(self.input_dim, self.latent_feature_dim)
        self.bn_1 = torch.nn.BatchNorm1d(self.latent_feature_dim)
        self.linear2 = torch.nn.Linear(self.latent_feature_dim, self.output_dim)

    def forward(self, x):
        # implemetation
        lf = self.linear1(x) # input_dim->latent_feature_dim
        x = self.bn_1(lf)
        x = F.relu(x)
        x = self.linear2(x) # latent_feature_dim->output_dim
        # x = F.softmax(x, dim=self.output_dim)
        
        return x, lf
    
if __name__ == "__main__":
    # chk
    model = Reg_Model(8, 3, 1)
    summary(model)

    # =================================================================
    # Layer (type:depth-idx)                   Param #
    # =================================================================
    # Reg_Model                                --
    # ├─Linear: 1-1                            27 -> input_dim * latent_dim + latent_dim = 8 * 3 + 3
    # ├─BatchNorm1d: 1-2                       6  
    # ├─Linear: 1-3                            4  -> latent_dim * output_dim + output_dim = 3 * 1  + 1
    # =================================================================
    # Total params: 37
    # Trainable params: 37
    # Non-trainable params: 0
    # =================================================================