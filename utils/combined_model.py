

#%%
import torch
from utils import *

class piped_classifier(torch.nn.Module):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x

# Define the combined model
# Define model1




#combined_model = piped_classifier(model1, model2)



#%%

