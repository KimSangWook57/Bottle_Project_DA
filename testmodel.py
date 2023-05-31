# model 관리 파일

import torch.nn as nn

# 모델 정의
# 모델 레이어 생성
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(3, 2)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(3, 2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(384)
        
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveMaxPool2d((6, 6))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool1(x)
        
        x = nn.functional.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool2(x)
        
        x = nn.functional.relu(self.conv3(x))
        x = self.batchnorm3(x)
        
        x = nn.functional.relu(self.conv4(x))
        x = self.batchnorm4(x)
        
        x = nn.functional.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

