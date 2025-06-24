
import torch.nn as nn



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(240*240*155, 10),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(10, 2)#,
            #nn.ReLU(),

            # nn.Dropout(p=0.4),
            # nn.Linear(300, 3)#,

            #nn.Softmax(dim=0)
        )

        self.to("cuda")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits