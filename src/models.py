# Models required for training Diffusion model
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, num_classes, save_path='model_weights.pth', patience=10, e_ratio=100, best_acc = 0):
        super(BaseModel, self).__init__()
        
        # Common attributes
        self.best_acc = best_acc
        self.save_path = save_path
        self.patience = patience
        self.e_ratio = e_ratio
        self.current_patience = 0
        self.best_epoch = 0
        
        # Common weight dictionaries
        self.weight_dic = {
            'train_loss': None,
            'train_acc': None,
            'val_acc': None,
            'val_loss': None
        }
        self.metrics_now = {
            'train_loss': None,
            'train_acc': None,
            'val_acc': None,
            'val_loss': None
        }
        self.metrics_best = {
            'train_loss': -np.inf,
            'train_acc': 0,
            'val_acc': 0,
            'val_loss': -np.inf
        }

    def early_stopping(self, thing, epoch):
        '''
        Incase you wanted to use best loss
        just use "-loss"
        '''
        self.check_weight()
        # Early stopping
        if (thing > self.best_acc) and (np.abs(thing - self.best_acc) > np.abs(self.best_acc) / self.e_ratio):
            self.best_acc = thing
            self.best_epoch = epoch
            self.current_patience = 0

            # Save the model's weights
            torch.save(self.state_dict(), self.save_path)
            print("<<<<<<<  !Model saved!  >>>>>>>")
            return False
        else:
            self.current_patience += 1
            # Check if the patience limit is reached
            if self.current_patience >= self.patience:
                print("Early stopping triggered!")
                return True
            else:
                return False
    
    def check_weight(self):
        for k in self.weight_dic.keys():
            if self.metrics_now[k] > self.metrics_best[k]:
                self.metrics_best[k] = self.metrics_now[k]
                self.weight_dic[k] = self.state_dict()

class FeatureExtractor(nn.Module):
    def __init__(self, drop=0.1, input_channels=1):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=128)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(drop)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=64)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(drop)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=16)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(drop)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(drop)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=2)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(drop)

        self.bnf = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))


        x = torch.flatten(x, 1)
        x = self.bnf(x)
        return x

class Classifier(BaseModel):
    def __init__(self, num_classes, drop=0.2, in_dim=1024, save_path='model_weights.pth', patience=10, e_ratio=100, best_acc = 0):
        # Initialize the base class
        super(Classifier, self).__init__(num_classes, save_path, patience, e_ratio, best_acc)
        
        # Model-specific layers
        self.fc1 = nn.Linear(in_dim, 128)
        self.dropout1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(drop)

        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(drop)  # Fixed: changed from dropout2 to dropout3

        self.fcc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        latent = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))  # This line seems unused in your original code
        x = F.softmax(self.fcc(latent), dim=1)
        return x

class Network(BaseModel):
    def __init__(self, num_classes, in_channels=2, save_path='model_weights.pth', patience=10, e_ratio=100, best_acc=0):
        # Initialize the base class
        super(Network, self).__init__(num_classes, save_path, patience, e_ratio, best_acc)
        
        # Model-specific attributes and layers
        self.in_ch = in_channels
        self.feature_extractor = FeatureExtractor(input_channels=in_channels)
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        if self.in_ch == 1:
            x = x.view(x.shape[0], 1, x.shape[1])  # Reshape input to (batch_size, channels, length)
        else:
            x = x.view(x.shape[0], x.shape[2], x.shape[1])
        features = self.feature_extractor(x)
        x = self.classifier(features)

        return x
    
class AuxNet(BaseModel):

    def __init__(
            self,
            n_layer = 4,
            in_dim = 1024*5,
            out_dim = 1024,
            num_classes = 26,
            include_y = False,
            best_acc = 0
    ):
        super().__init__(num_classes, best_acc=best_acc)
        layers = []
        hidden_size = int((in_dim+out_dim)/2)
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, out_dim))
        layers.append(nn.BatchNorm1d(1024))
        
        self.model = nn.Sequential(*layers)

        
        self.label_coder = lambda y:F.one_hot(y, num_classes=num_classes)
        self.include_y = include_y


    def forward(self, x):
        return self.model(x)

    


