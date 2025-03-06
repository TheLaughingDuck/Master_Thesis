from torch import nn
import numpy as np
from monai.networks.nets import SwinUNETR
import torch

# Define Classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768*4**3, 768*32),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(768*32, 768*16),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(768*16, 768*8),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(768*8, 768),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(768, 3),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Detective_Classifier(nn.Module):
    '''
    A dummy classifier that is simple and quick so that it can be used for debugging the training loop.'''
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768*4**3, 3),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



# Define Feature Extractinator
class EmbedSwinUNETR(SwinUNETR):
    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        # Just return this embedding
        return(dec4)
        
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0, enc0)
        # logits = self.out(out)
        # return logits


class Detective_EmbedSwinUNETR(SwinUNETR):
    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        
        return(torch.randn([1,768,4,4,4]))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)