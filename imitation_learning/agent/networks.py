import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, n_classes=3, history_length=1): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural networkp
        self.model = nn.Sequential(
            nn.Conv2d(history_length, 16, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # output: 16 x 45 x 45

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),          # output: 32 x 21 x 21

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),          # output: 64 x 9 x 9

            nn.Flatten(), 
            nn.Linear(64*9*9, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        

    def forward(self, x):
        # TODO: compute forward pass
        return self.model(x)
