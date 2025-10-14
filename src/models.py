# Models required for training Diffusion model
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseModel(nn.Module):
    def __init__(self, save_path='model_weights.pth', patience=10, e_ratio=100, best_acc = 0):
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
            'test_acc': None,
            'test_loss': None
        }
        self.metrics_now = {
            'train_loss': None,
            'train_acc': None,
            'test_acc': None,
            'test_loss': None
        }
        self.metrics_best = {
            'train_loss': -100,
            'train_acc': 0,
            'test_acc': 0,
            'test_loss': -100
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

        # self.bnf = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))


        x = torch.flatten(x, 1)
        # x = self.bnf(x)
        return x

class Classifier(BaseModel):
    def __init__(self, num_classes, drop=0.2, in_dim=1024, save_path='model_weights.pth', patience=10, e_ratio=100, best_acc = 0):
        # Initialize the base class
        super(Classifier, self).__init__(save_path, patience, e_ratio, best_acc)
        
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
    def __init__(self, num_classes=26, in_channels=2, save_path='model_weights.pth', patience=10, e_ratio=100, best_acc=0):
        # Initialize the base class
        super(Network, self).__init__(save_path, patience, e_ratio, best_acc)
        
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
        super().__init__(best_acc=best_acc)
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

class DiffusionProcess:

    def __init__(self, T, beta_start, beta_end):

        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    # Forward diffusion process
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0, device=device)
        sqrt_ab = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(0)
        sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(0)

        try:
            return sqrt_ab.T * x0 + sqrt_one_minus_ab.T * noise, noise
        except RuntimeError:
            return sqrt_ab.T.unsqueeze(2) * x0 + sqrt_one_minus_ab.T.unsqueeze(2) * noise, noise

    # Backward diffusion process
    def p_sample(self, x_t, t, noise_pred):
        beta_t = self.beta[t].unsqueeze(0)
        alpha_t = self.alpha[t].unsqueeze(0)
        alpha_bar_t = self.alpha_cumprod[t].unsqueeze(0)

        mu = (1/torch.sqrt(alpha_t)).T * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)).T * noise_pred
        )

        if t[0] > 0:
            z = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_prev = mu + sigma.T * z
        else:
            x_prev = mu
            
        return x_prev

def sinusoidal_embedding(timesteps, dim):
    
    device = timesteps.device
    half_dim = dim // 2
    freq = torch.exp(
        -torch.arange(half_dim, device=device) * torch.log(torch.tensor(10000.0)) / half_dim
    )
    angles = timesteps[:, None].float() * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb/5  # (batch, dim)

class ConvBlock(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, ks=3, pad=1, drop=None, mp=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=ks, padding=pad)
        # self.bn = nn.BatchNorm1d(out_channel)

        # if drop is not None:
        #     self.dropout = nn.Dropout(drop)
        # else:
        #     self.dropout = None

        # if mp is not None:
        #     self.pool = nn.MaxPool1d(kernel_size=mp)
        # else:
        #     self.pool = None

    def forward(self,x):
        x = self.conv(x)
        # x = self.bn(x)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # if self.pool is not None:
        #     x = self.pool(x)

        return x

class Down(nn.Module):

    def __init__(self, mp=2):
        super().__init__()
        self.mp = nn.MaxPool1d(mp,mp)

    def forward(self,x):
        x_d = self.mp(x)

        return x_d
    
class Up(nn.Module):

    def __init__(self,in_channel=1, out_channel=1, us=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=us, stride=us)

    def forward(self, x_u, x):
        x_u = self.up(x_u)
        x = torch.concat([x_u,x], dim=1)

        return x

class ConvEmbed(nn.Module):

    def __init__(self, in_channel=1, out_channel=None, mp=None,drop=None, ks=3, pad=1, activation='relu'):
        super().__init__()

        self.conv1 = ConvBlock(in_channel, in_channel, ks=ks, pad=pad, mp=mp, drop=drop)
        if out_channel is not None:
            self.conv2 = ConvBlock(in_channel,out_channel, ks=ks, pad=pad, mp=mp, drop=drop)
        else:
            self.conv2 = ConvBlock(in_channel,in_channel, ks=ks, pad=pad, mp=mp, drop=drop)

        if activation == 'relu':
            self.af = nn.ReLU()
        elif activation == 'silu':
            self.af = nn.SiLU()
        elif activation == 'gelu':
            self.af = nn.GELU()

    def forward(self,x):
        x = self.af(self.conv1(x)) + x
        x = self.af(self.conv2(x))

        return x
    
class Conv1(nn.Module):

    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel,out_channel,kernel_size=1,stride=1,padding=0) 

    def forward(self,x):
        return self.conv(x)


class UNET(BaseModel): 

    def __init__(
            self,
            in_channel_z = 1,
            in_channel_x = 2,
            out_channel_z = 1,
    ):
        super().__init__()

        self.add_time = lambda x, t: x + sinusoidal_embedding(t, x.shape[2]).unsqueeze(1)

        self.zd_1 = ConvEmbed(64,activation='relu')
        self.zd_2 = ConvEmbed(128,activation='relu')
        self.zd_3 = ConvEmbed(256,activation='relu')
        self.zd_4 = ConvEmbed(512,activation='relu')
        self.zd_5 = ConvEmbed(1024,activation='relu')

        self.xd_1 = ConvEmbed(64,activation='relu')
        self.xd_2 = ConvEmbed(128,activation='relu')
        self.xd_3 = ConvEmbed(256,activation='relu')
        self.xd_4 = ConvEmbed(512,activation='relu')
        self.xd_5 = ConvEmbed(1024,activation='relu')

        self.d4 = Down(4)
        self.d2 = Down(2)

        self.zuu_1 = Up(256,64,4)
        self.zuu_2 = Up(512,128,4)
        self.zuu_3 = Up(1024,256,2)
        self.zuu_4 = Up(1024,512,2)
        self.zu_1 = ConvEmbed(128,activation='relu')
        self.zu_2 = ConvEmbed(256,activation='relu')
        self.zu_3 = ConvEmbed(512,activation='relu')
        self.zu_4 = ConvEmbed(1024,activation='relu')

        self.xuu_1 = Up(512,64,4)
        self.xuu_2 = Up(1024,128,4)
        self.xuu_3 = Up(2048,256,2)
        self.xuu_4 = Up(1024,512,2)
        self.xu_1 = ConvEmbed(256,activation='relu')
        self.xu_2 = ConvEmbed(512,activation='relu')
        self.xu_3 = ConvEmbed(1024,activation='relu')
        self.xu_4 = ConvEmbed(2048,activation='relu')

        # self.conv_x = Conv1(256,128)

        self.conv0 = Conv1(384,192)
        # self.convf = ConvEmbed(64,64)
        # self.conv2 = Conv1(64,out_channel_z)

        self.conv1 = Conv1(192,192)
        

    def forward(self,x,z,t):
        
        # Encoder
        z1 = z.repeat(1,64//2,1)
        x1 = x.repeat(1,64//x.shape[1],1)

        z1 = self.zd_1(z1)
        x1 = self.xd_1(x1)

        z2 = self.d4(z1)
        x2 = self.d4(x1)
        x2 = torch.concat([x2,z2],dim=1)
        z2 = z2.repeat(1,2,1)
        z2 = self.zd_2(z2)
        x2 = self.xd_2(x2)

        z3 = self.d4(z2)
        x3 = self.d4(x2)
        x3 = torch.concat([x3,z3],dim=1)
        z3 = z3.repeat(1,2,1)
        z3 = self.zd_3(z3)
        x3 = self.xd_3(x3)

        z4 = self.d2(z3)
        x4 = self.d2(x3)
        x4 = torch.concat([x4,z4],dim=1)
        z4 = z4.repeat(1,2,1)
        z4 = self.zd_4(z4)
        x4 = self.xd_4(x4)

        z5 = self.d2(z4)
        x5 = self.d2(x4)
        x5 = torch.concat([x5,z5],dim=1)
        z5 = z5.repeat(1,2,1)
        z5 = self.zd_5(z5)
        x5 = self.xd_5(x5)

        # Decoder
        z4_u = self.zuu_4(z5,z4)
        x4_u = self.xuu_4(x5,torch.concat([x4,z4_u],dim=1))
        z4_u = self.zu_4(z4_u)
        x4_u = self.xu_4(x4_u)

        z3_u = self.zuu_3(z4_u,z3)
        x3_u = self.xuu_3(x4_u,torch.concat([x3,z3_u],dim=1))
        z3_u = self.zu_3(z3_u)
        x3_u = self.xu_3(x3_u)

        z2_u = self.zuu_2(z3_u,z2)
        x2_u = self.xuu_2(x3_u,torch.concat([x2,z2_u],dim=1))
        z2_u = self.zu_2(z2_u)
        x2_u = self.xu_2(x2_u)

        z1_u = self.zuu_1(z2_u,z1)
        x1_u = self.xuu_1(x2_u,torch.concat([x1,z1_u],dim=1))
        z1_u = self.zu_1(z1_u)
        x1_u = self.xu_1(x1_u)

        # Final flow
        # x_f = self.conv_x(x1_u)
        
        # z_f = torch.concat([z1_u,x_f],dim=1)
        z_f = torch.concat([z1_u,x1_u],dim=1)
        z_f = self.conv0(z_f)
        z_f = torch.concat([self.conv1(z_f),z_f], dim=1)
        # z_f = self.convf(z_f)
        # z_f = self.conv2(z_f)


        return z_f







