'''
Module for defining the classifier.
'''

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_channels)
        )
    
    def forward(self, x):
        return self.block(x)


class Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Classifier, self).__init__()

        self.ConvSequence = nn.Sequential(
            ConvBlock(2, 2, (10,10,10)),
            ConvBlock(2, 2, (10,10,10)),
            ConvBlock(2, 2, (10,10,10)),
            ConvBlock(2, 2, (8,8,8)),
            ConvBlock(2, 2, (8,8,8)),
            ConvBlock(2, 2, (8,8,8)),
            ConvBlock(2, 2, (8,8,8)),
            ConvBlock(2, 2, (8,8,8)),
            ConvBlock(2, 2, (8,8,8)),
            ConvBlock(2, 1, (8,8,8)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (5,5,5)),
            ConvBlock(1, 1, (3,3,3)),
            ConvBlock(1, 1, (3,3,3)),
            ConvBlock(1, 1, (3,3,3)),
            ConvBlock(1, 1, (3,3,3))
        )
        # self.net = nn.Sequential(a)

        # TESTING
        self.flatten = nn.Flatten()
        self.ConvBlock = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=2, kernel_size=(10,10,10)),
            nn.ReLU()
        )
        self.lin_rel_stack = nn.Sequential(
            nn.Linear(12**3, 10),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(10, 3)
        )

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameter count: {format(n_params, ",").replace(",", ".")} \N{Abacus}.\n")

        # Put the model on the gpu. This should be at the end of the class __init__
        self.device = "cuda"
        self.to(self.device)
        print(f"Model uses {self.device} device.")
    
    def forward(self, x):
        x = self.ConvSequence(x)
        #print(f"x shape: {x.shape}")
        x = self.flatten(x)
        logits = self.lin_rel_stack(x)
        return(logits)
        
        return self.net(x)


# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchsummary import summary

# class ResNet50_3D(nn.Module):
#     def __init__(self, num_classes=3):
#         super(ResNet50_3D, self).__init__()
        
#         # Load pre-trained ResNet-50 model
#         resnet50 = models.resnet50(pretrained=True)
        
#         # Modify the first layer to accept 3D input (Conv3D)
#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        
#         # Replace ResNet's original conv1 layer with the new 3D conv1
#         self.conv1.weight.data = resnet50.conv1.weight.data  # You can initialize the weights or randomize them
        
#         # Replace the 2D layers in ResNet with 3D layers
#         self.layer1 = resnet50.layer1
#         self.layer2 = resnet50.layer2
#         self.layer3 = resnet50.layer3
#         self.layer4 = resnet50.layer4
        
#         # Modify the final fully connected layer to output 3 logits (instead of 1000 for ImageNet)
#         self.fc = nn.Linear(2048, num_classes)  # ResNet50's last layer before fc has 2048 features
        
#     def forward(self, x):
#         # Pass the input through the network
#         x = self.conv1(x)  # 3D convolution for input
#         x = self.bn1(x)  # BatchNorm layer
#         x = self.relu(x)  # ReLU activation
#         x = self.maxpool(x)  # Max pooling
        
#         # Pass through the remaining layers
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         # Average pooling
#         x = self.avgpool(x)
        
#         # Flatten and pass through the fully connected layer
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
        
#         return x

# # Define input tensor shape (batch_size, channels, depth, height, width)
# input_tensor = torch.randn(1, 1, 240, 240, 155)  # Example 3D input
# model = ResNet50_3D(num_classes=3)

# # Print model summary
# summary(model, input_size=(1, 240, 240, 155))

# # Example forward pass
# output = model(input_tensor)
# print(output.shape)  # Expected output: (1, 3) since there are 3 logits
