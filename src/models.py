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

    def __init__(self, T, beta_start, beta_end, beta_type = 'lin'):
        
        if beta_type == 'log':
            self.beta = torch.logspace(beta_start, beta_end, T, device=device)
        elif beta_type == 'lin':
            beta_start = np.exp(beta_start)
            beta_end = np.exp(beta_end)
            self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    # Forward diffusion process
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0, device=device)
        sqrt_ab = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(0)
        sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(0)

        return sqrt_ab.T.unsqueeze(1) * x0 + sqrt_one_minus_ab.T.unsqueeze(1) * noise, noise

    # Backward diffusion process
    def p_sample(self, x_t, t, noise_pred):
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_cumprod[t]
        
        # Ensure these are the right shape for broadcasting
        beta_t = beta_t.view(-1, 1, 1)
        alpha_t = alpha_t.view(-1, 1, 1)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)
        
        # Calculate the mean
        mu = (1/torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )

        if t[0] > 0:
            z = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_prev = mu + sigma * z
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
    return emb  # (batch, dim)

class ConvBlock(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, ks=3, pad=1, drop=None, mp=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=ks, padding=pad)
        self.bn = nn.BatchNorm1d(out_channel)
        if drop is not None:
            self.dropout = nn.Dropout(drop)
        else:
            self.dropout = None
        if mp is not None:
            self.pool = nn.MaxPool1d(kernel_size=mp)
        else:
            self.pool = None

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.pool is not None:
            x = self.pool(x)

        return x

class Down(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, mp=2):
        super().__init__()
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)
        self.mp = nn.MaxPool1d(mp,mp)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_d = self.mp(x)

        return x, x_d
    
class Up(nn.Module):

    def __init__(self,in_channel=1, out_channel=1, us=2, ch_d=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channel, in_channel//ch_d, kernel_size=us, stride=us)
        self.up1 = nn.ConvTranspose1d(in_channel, in_channel//2, kernel_size=us, stride=us)
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)

    def forward(self, x_u, x):
        if x is not None:
            x_u = self.up(x_u)
            x = torch.concat([x_u,x], dim=1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x)) 
        else:
            x_u = self.up1(x_u)
            x = torch.concat([x_u,x_u], dim=1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x)) 

        return x

class ConvEmbed(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, last_layer=False, mp=None,drop=None, ks=3, pad=1):
        super().__init__()

        self.conv1 = ConvBlock(in_channel, out_channel, ks=ks, pad=pad, mp=mp, drop=drop)
        self.conv2 = ConvBlock(out_channel,out_channel, ks=ks, pad=pad, mp=mp, drop=drop)

        self.last_layer = last_layer

        self.ch_keep = True if in_channel==out_channel else False

    def forward(self,x):

        if self.ch_keep:

            x = F.relu(self.conv1(x)) + x
            if not self.last_layer:
                x = F.relu(self.conv2(x)) + x
            else:
                x = self.conv2(x)
        else:
            
            x = F.relu(self.conv1(x))
            if not self.last_layer:
                x = F.relu(self.conv2(x)) 
            else:
                x = self.conv2(x)

        return x

class UNET(BaseModel): 

    def __init__(
            self,
            in_channel_z = 1,
            in_channel_x = 2,
            out_channel_z = 1,
    ):
        super().__init__()

        self.d1_z = Down(in_channel_z, 64, mp=4) # 1024
        self.d2_z = Down(64, 128, mp=4) # 256
        self.d3_z = Down(128, 256) # 64
        self.d4_z = Down(256, 512) # 32

        self.d1_x = Down(in_channel_x, 64, mp=4)
        self.d2_x = Down(64, 128, mp = 4)
        self.d3_x = Down(128, 256)
        self.d4_x = Down(256, 512)

        self.kz_0 = ConvEmbed(512, 1024)
        self.kx_0 = ConvEmbed(1024, 2048)
        self.kz = ConvEmbed(1024, 1024)
        self.kx = ConvEmbed(2048, 1024)

        # self.u4_z = Up(1024,512,ch_d=2)
        self.u3_z = Up(1024,512,ch_d=2)
        self.u2_z = Up(512,256,ch_d=2)
        self.u1_z = Up(256,128, us=4,ch_d=2)
        self.u0_z = Up(128,64, us=4,ch_d=2)

        # self.u4_x = Up(2048,512,ch_d=4)
        self.u3_x = Up(1024,512,ch_d=2)
        self.u2_x = Up(512,256,ch_d=2)
        self.u1_x = Up(256,128, us=4,ch_d=2)
        self.u0_x = Up(128,64, us=4,ch_d=2)

        self.f1 = ConvEmbed(128,64)

        self.f2 = ConvEmbed(64,16)

        self.conv1 = nn.Conv1d(16, out_channel_z, kernel_size=1, stride=1, padding=0)

        

    def forward(self,x,z,t):

        x = x + sinusoidal_embedding(t, x.shape[2]).unsqueeze(1)
        z = z + sinusoidal_embedding(t, z.shape[2]).unsqueeze(1)
        # Downsampling Z
        z1, z1_d = self.d1_z(z)
        # z1 = z1 + sinusoidal_embedding(t, z1.shape[2]).unsqueeze(1)
        # z1_d = z1_d + sinusoidal_embedding(t, z1_d.shape[2]).unsqueeze(1)

        z2, z2_d = self.d2_z(z1_d)
        z2 = z2 + sinusoidal_embedding(t, z2.shape[2]).unsqueeze(1)
        z2_d = z2_d + sinusoidal_embedding(t, z2_d.shape[2]).unsqueeze(1)

        z3, z3_d = self.d3_z(z2_d)
        # z3 = z3 + sinusoidal_embedding(t, z3.shape[2]).unsqueeze(1)
        # z3_d = z3_d + sinusoidal_embedding(t, z3_d.shape[2]).unsqueeze(1)

        z4, z4_d = self.d4_z(z3_d)
        z4 = z4 + sinusoidal_embedding(t, z4.shape[2]).unsqueeze(1)
        z4_d = z4_d + sinusoidal_embedding(t, z4_d.shape[2]).unsqueeze(1)

        # Downsampilng X
        x1, x1_d = self.d1_x(x)
        # x1 = x1 + sinusoidal_embedding(t, x1.shape[2]).unsqueeze(1)
        # x1_d = x1_d + sinusoidal_embedding(t, x1_d.shape[2]).unsqueeze(1)

        x2, x2_d = self.d2_x(x1_d)
        x2 = x2 + sinusoidal_embedding(t, x2.shape[2]).unsqueeze(1)
        x2_d = x2_d + sinusoidal_embedding(t, x2_d.shape[2]).unsqueeze(1)

        x3, x3_d = self.d3_x(x2_d)
        # x3 = x3 + sinusoidal_embedding(t, x3.shape[2]).unsqueeze(1)
        # x3_d = x3_d + sinusoidal_embedding(t, x3_d.shape[2]). unsqueeze(1)
        
        x4, x4_d = self.d4_x(x3_d)
        x4 = x4 + sinusoidal_embedding(t, x4.shape[2]).unsqueeze(1)
        x4_d = x4_d + sinusoidal_embedding(t, x4_d.shape[2]). unsqueeze(1) # x5_d = x5_d + sinusoidal_embedding(t, x5_d.shape[2]). unsqueeze(1)              
        # Concat Low
        z5 = z4_d
        x5 = torch.concat([x4_d, z5], dim=1)

        # Process Low
        z5 = self.kz_0(z5)
        x5 = self.kx_0(x5)

        z7 = self.kz(z5)
        z7 = z7 + sinusoidal_embedding(t, z7.shape[2]).unsqueeze(1)
        x7 = self.kx(x5)
        x7 = x7 + sinusoidal_embedding(t, x7.shape[2]).unsqueeze(1)

        # Upsampling Z
        # z4_u = self.u4_z(z7, None)
        # z4_u = z4_u + sinusoidal_embedding(t, z4_u.shape[2]).unsqueeze(1)

        z3_u = self.u3_z(z7, z4)
        z3_u = z3_u + sinusoidal_embedding(t, z3_u.shape[2]).unsqueeze(1)

        z2_u = self.u2_z(z3_u, z3)
        # z2_u = z2_u + sinusoidal_embedding(t, z2_u.shape[2]).unsqueeze(1)

        z1_u = self.u1_z(z2_u, z2)
        z1_u = z1_u + sinusoidal_embedding(t, z1_u.shape[2]).unsqueeze(1) 


        z0_u = self.u0_z(z1_u, z1)
        z0_u = z0_u + sinusoidal_embedding(t, z0_u.shape[2]).unsqueeze(1) 


        # Upsampling X
        # x4_u = self.u4_x(x7, x4)
        # x4_u = x4_u + sinusoidal_embedding(t, x4_u.shape[2]).unsqueeze(1)

        x3_u = self.u3_x(x7, x4)
        x3_u = x3_u + sinusoidal_embedding(t, x3_u.shape[2]).unsqueeze(1)

        x2_u = self.u2_x(x3_u, x3)
        # x2_u = x2_u + sinusoidal_embedding(t, x2_u.shape[2]).unsqueeze(1)

        x1_u = self.u1_x(x2_u, x2)
        x1_u = x1_u + sinusoidal_embedding(t, x1_u.shape[2]).unsqueeze(1)

        x0_u = self.u0_x(x1_u, x1)
        x0_u = x0_u + sinusoidal_embedding(t, x0_u.shape[2]).unsqueeze(1)

        # Concat High
        z_f = torch.concat([z0_u, x0_u], dim=1)
        # z_f = z_f + sinusoidal_embedding(t, z_f.shape[2]).unsqueeze(1)

        # B_3ranck Processing
        z_f = self.f1(z_f)
        z_f = self.f2(z_f)

        z_f = self.conv1(z_f)


        return z_f


class UNETAE(BaseModel): 

    def __init__(
            self,
            in_channel_z = 1,
            out_channel_z = 1,
    ):
        super().__init__()

        i = 2
        self.d1_z = Down(in_channel_z, 64, mp=4) # 1024
        self.d2_z = Down(64, 128) # 256
        self.d2_z_0 = Down(64*2, 128*i) # 256
        self.d3_z = Down(128*i, 256*i) # 64
        self.d4_z = Down(256*i, 512*i) # 32

        self.kz_0 = ConvEmbed(512*i, 1024*i)
        self.kz = ConvEmbed(1024*i, 1024*i)

        # self.u4_z = Up(1024,512,ch_d=2)
        self.u4_z = Up(1024*i,512*i,ch_d=2)
        self.u3_z = Up(512*i,256*i,ch_d=2)
        self.u2_z_0 = Up(256*i,128*2, ch_d=2)
        self.u2_z = Up(256,128, ch_d=2)
        self.u1_z = Up(128,128, us=4,ch_d=2)

        self.f1 = ConvEmbed(128,128) #skip it

        self.f2 = ConvEmbed(128,128)
        self.f20 = ConvEmbed(128,128)
        self.f21 = ConvEmbed(128,128)

        self.f3 = ConvEmbed(128,64)

        self.conv1 = nn.Conv1d(64, out_channel_z, kernel_size=1, stride=1, padding=0)

        

    def forward(self,x,t=None):

        if t is None:
            x1, x1_d = self.d1_z(x)
            x2, x2_d = self.d2_z(x1_d)
            x2_0, x2_d_0 = self.d2_z_0(x2_d)
            x3, x3_d = self.d3_z(x2_d_0)
            x4, x4_d = self.d4_z(x3_d)

            x5 = self.kz_0(x4_d)
            x5 = self.kz(x5)

            x4_u = self.u4_z(x5, x4)
            x3_u = self.u3_z(x4_u,x3)
            x2_u_0 = self.u2_z_0(x3_u,x2_0)
            x2_u = self.u2_z(x2_u_0,x2)
            x1_u = self.u1_z(x2_u,x1)

            x = self.f1(x1_u)
            x = self.f2(x)
            x = self.f20(x)
            x = self.f21(x)
            x = self.f3(x)
            x = self.conv1(x)
        else:
            x1, x1_d = self.d1_z(x)
            z1 = x1 + sinusoidal_embedding(t, x1.shape[2]).unsqueeze(1)
            x1_d = x1_d + sinusoidal_embedding(t, x1_d.shape[2]).unsqueeze(1)

            x2, x2_d = self.d2_z(x1_d)
            x2 = x2 + sinusoidal_embedding(t, x2.shape[2]).unsqueeze(1)
            x2_d = x2_d + sinusoidal_embedding(t, x2_d.shape[2]).unsqueeze(1)
            
            x3, x3_d = self.d3_z(x2_d)
            x3 = x3 + sinusoidal_embedding(t, x3.shape[2]).unsqueeze(1)
            x3_d = x3_d + sinusoidal_embedding(t, x3_d.shape[2]).unsqueeze(1)
            
            x4, x4_d = self.d4_z(x3_d)
            x4 = x4 + sinusoidal_embedding(t, x4.shape[2]).unsqueeze(1)
            x4_d = x4_d + sinusoidal_embedding(t, x4_d.shape[2]).unsqueeze(1)
            

            x5 = self.kz_0(x4_d)
            x5 = self.kz(x5)
            x5 = x5 + sinusoidal_embedding(t,x5.shape[2]).unsqueeze(1)

            x4_u = self.u4_z(x5, x4)
            x4_u = x4_u + sinusoidal_embedding(t,x4_u.shape[2]).unsqueeze(1)

            x3_u = self.u3_z(x4_u,x3)
            x3_u = x3_u + sinusoidal_embedding(t,x3_u.shape[2]).unsqueeze(1)

            x2_u = self.u2_z(x3_u,x2)
            x2_u = x2_u + sinusoidal_embedding(t,x2_u.shape[2]).unsqueeze(1)

            x1_u = self.u1_z(x2_u,x1)
            x1_u = x1_u + sinusoidal_embedding(t,x1_u.shape[2]).unsqueeze(1)

            x = self.f1(x1_u)
            x = self.f2(x)
            x = self.conv1(x)

        

        return x


class FEBased(BaseModel):
    def __init__(self, drop=0.1, input_channels=1):
        super().__init__()

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

        self.conv5 = nn.Conv1d(128, 512, kernel_size=2)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(drop)


        self.embed1 = ConvEmbed(16,16)
        self.embed2 = ConvEmbed(32,32)
        self.embed3 = ConvEmbed(64,64)
        self.embed4 = ConvEmbed(128,128)
        # self.embed5 = ConvEmbed(512,512)

        # self.bnf = nn.BatchNorm1d(1024)

    def forward(self, x, t):

        def with_time(x):
            x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
            x = self.embed1(x)
            x = x + sinusoidal_embedding(t,x.shape[2]).unsqueeze(1)

            x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
            x = self.embed2(x)
            x = x + sinusoidal_embedding(t,x.shape[2]).unsqueeze(1)
            
            x = self.pool3(self.dropout3(F.relu(self.bn3(self.conv3(x)))))
            x = self.embed3(x)
            x = x + sinusoidal_embedding(t,x.shape[2]).unsqueeze(1)
            
            x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
            x = self.embed4(x)
            # x = x + sinusoidal_embedding(t,x.shape[2]).unsqueeze(1)
            
            x = self.conv5(x)


        def without(x):
            x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
            x = self.embed1(x)

            x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
            x = self.embed2(x)
            
            x = self.pool3(self.dropout3(F.relu(self.bn3(self.conv3(x)))))
            x = self.embed3(x)
            
            x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
            x = self.embed4(x)
            # x = x + sinusoidal_embedding(t,x.shape[2]).unsqueeze(1)
            
            x = self.conv5(x)
            return x

        if t is None:
            x = without(x)
        else :
            x = with_time(x)

        x = x.reshape(-1,2,1024)
        # x = self.bnf(x)
        return x