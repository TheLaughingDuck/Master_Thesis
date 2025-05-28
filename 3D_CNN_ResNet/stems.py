import torch.nn as nn
from torch.nn import Sequential


class modifybasicstem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    '''
    Equivalent to https://github.com/farazahmeds/Classification-of-brain-tumor-using-Spatiotemporal-models/blob/main/stems/resnet_mixed_conv.py
    with the change of letting the number of channels be adjustable.
    '''

    def __init__(self, channels: int = 1):
        super(modifybasicstem, self).__init__(
            nn.Conv3d(
                channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem4MRI(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    '''
    Equivalent to https://github.com/farazahmeds/Classification-of-brain-tumor-using-Spatiotemporal-models/blob/main/stems/resnet2p1.py
    with the change of letting the number of channels be adjustable.
    '''
    
    def __init__(self, channels: int = 1):
        super(R2Plus1dStem4MRI, self).__init__(
            nn.Conv3d(
                channels,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
