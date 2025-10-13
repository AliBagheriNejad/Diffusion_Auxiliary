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

        return sqrt_ab.T * x0 + sqrt_one_minus_ab.T * noise, noise

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

    def __init__(self,in_channel=1, out_channel=1, us=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channel, in_channel//2, kernel_size=us, stride=us)
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)
        # self.conv3 = ConvBlock(out_channel, out_channel)
        # self.conv4 = ConvBlock(out_channel, out_channel)
        # self.conv5 = ConvBlock(out_channel, out_channel)
        # self.conv6 = ConvBlock(out_channel, out_channel)

    def forward(self, x_u, x):
        x_u = self.up(x_u)
        if x.shape[1] != x_u.shape[1]:
            n_r = x_u.shape[1] // x.shape[1]
            x = x.repeat(1,n_r,1)
        x = torch.concat([x_u,x], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))

        return x

class ConvEmbed(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, last_layer=False, mp=None,drop=None, ks=3, pad=1):
        super().__init__()

        self.conv2 = ConvBlock(out_channel,out_channel, ks=ks, pad=pad, mp=mp, drop=drop)
        self.conv1 = ConvBlock(in_channel, out_channel, ks=ks, pad=pad, mp=mp, drop=drop)

        self.last_layer = last_layer

    def forward(self,x):
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

        self.d1_z = Down(in_channel_z, 64)
        self.d2_z = Down(64, 128)
        self.d3_z = Down(128, 256)
        self.d4_z = Down(256, 256)
        self.d5_z = Down(256, 256)
        self.d6_z = Down(256, 256)

        self.d1_x = Down(in_channel_x, 64)
        self.d2_x = Down(64, 128)
        self.d3_x = Down(128, 256)
        self.d4_x = Down(256, 256)
        self.d5_x = Down(256, 256)
        self.d6_x = Down(256, 256)

        self.kz = ConvEmbed(256, 512)
        self.kx = ConvEmbed(512,512)

        self.u6_z = Up(512,512)
        self.u5_z = Up(512,512)
        self.u4_z = Up(512,512)
        self.u3_z = Up(512,512)
        self.u2_z = Up(512,512)
        self.u1_z = Up(512,512)
        
        self.u6_x = Up(512,512)
        self.u5_x = Up(512,512)
        self.u4_x = Up(512,512)
        self.u3_x = Up(512,512)
        self.u2_x = Up(512,512)
        self.u1_x = Up(512,512)

        self.f1_3 = ConvEmbed(1024,256)
        # self.f2_3 = ConvEmbed(256,512)
        # self.f6_3 = ConvEmbed(512,512)
        # self.f7_3 = ConvEmbed(512,512)
        # self.f8_3 = ConvEmbed(512,512)
        # self.f9_3 = ConvEmbed(512,512)
        # self.f3_3 = ConvEmbed(512,256)
        # self.f4_3 = ConvEmbed(256,256)
        self.f5_3 = ConvEmbed(256,out_channel_z, True)

        # self.f1_5 = ConvEmbed(128,256, ks=5, pad=2)
        # self.f2_5 = ConvEmbed(256,512, ks=5, pad=2)
        # self.f3_5 = ConvEmbed(512,256, ks=5, pad=2)
        # self.f4_5 = ConvEmbed(256,128, ks=5, pad=2)
        # self.f5_5 = ConvEmbed(128,out_channel_z, True, ks=5, pad=2)

        # self.f1_7 = ConvEmbed(128,256, ks=7, pad=3)
        # self.f2_7 = ConvEmbed(256,512, ks=7, pad=3)
        # self.f3_7 = ConvEmbed(512,256, ks=7, pad=3)
        # self.f4_7 = ConvEmbed(256,128, ks=7, pad=3)
        # self.f5_7 = ConvEmbed(128,out_channel_z, True, ks=7, pad=3)

        # self.final = ConvEmbed(3*out_channel_z, out_channel_z, True)

        # self.fc = nn.Sequential(
        #     nn.Linear(64*out_channel_z,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,1024)
        # )

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

        z5, z5_d = self.d5_z(z4_d)
        # z5 = z5 + sinusoidal_embedding(t, z5.shape[2]).unsqueeze(1)
        # z5_d = z5_d + sinusoidal_embedding(t, z5_d.shape[2]).unsqueeze(1)

        z6, z6_d = self.d6_z(z5_d)
        z6 = z6 + sinusoidal_embedding(t, z6.shape[2]).unsqueeze(1)
        z6_d = z6_d + sinusoidal_embedding(t, z6_d.shape[2]).unsqueeze(1)

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
        x4_d = x4_d + sinusoidal_embedding(t, x4_d.shape[2]). unsqueeze(1)

        x5, x5_d = self.d5_x(x4_d)
        # x5 = x5 + sinusoidal_embedding(t, x5.shape[2]).unsqueeze(1)
        # x5_d = x5_d + sinusoidal_embedding(t, x5_d.shape[2]). unsqueeze(1)
        
        x6, x6_d = self.d6_x(x5_d)
        x6 = x6 + sinusoidal_embedding(t, x6.shape[2]).unsqueeze(1)
        x6_d = x6_d + sinusoidal_embedding(t, x6_d.shape[2]). unsqueeze(1)

        # Concat Low
        z7 = z6_d
        x7 = torch.concat([x6_d, z7], dim=1)

        # Process Low
        z7 = self.kz(z7)
        z7 = z7 + sinusoidal_embedding(t, z7.shape[2]).unsqueeze(1)
        x7 = self.kx(x7)
        x7 = x7 + sinusoidal_embedding(t, x7.shape[2]).unsqueeze(1)

        # Upsampling Z
        z6_u = self.u6_z(z7, z6)
        # z6_u = z6_u + sinusoidal_embedding(t, z6_u.shape[2]).unsqueeze(1)

        z5_u = self.u5_z(z6_u, z5)
        z5_u = z5_u + sinusoidal_embedding(t, z5_u.shape[2]).unsqueeze(1)

        z4_u = self.u4_z(z5_u, z4)
        # z4_u = z4_u + sinusoidal_embedding(t, z4_u.shape[2]).unsqueeze(1)

        z3_u = self.u3_z(z4_u, z3)
        z3_u = z3_u + sinusoidal_embedding(t, z3_u.shape[2]).unsqueeze(1)

        z2_u = self.u2_z(z3_u, z2)
        # z2_u = z2_u + sinusoidal_embedding(t, z2_u.shape[2]).unsqueeze(1)

        z1_u = self.u1_z(z2_u, z1)
        z1_u = z1_u + sinusoidal_embedding(t, z1_u.shape[2]).unsqueeze(1) 

        # Upsampling X
        x6_u = self.u6_x(x7, x6)
        # x6_u = x6_u + sinusoidal_embedding(t, x6_u.shape[2]).unsqueeze(1)

        x5_u = self.u5_x(x6_u, x5)
        x5_u = x5_u + sinusoidal_embedding(t, x5_u.shape[2]).unsqueeze(1)

        x4_u = self.u4_x(x5_u, x4)
        # x4_u = x4_u + sinusoidal_embedding(t, x4_u.shape[2]).unsqueeze(1)

        x3_u = self.u3_x(x4_u, x3)
        x3_u = x3_u + sinusoidal_embedding(t, x3_u.shape[2]).unsqueeze(1)

        x2_u = self.u2_x(x3_u, x2)
        # x2_u = x2_u + sinusoidal_embedding(t, x2_u.shape[2]).unsqueeze(1)

        x1_u = self.u1_x(x2_u, x1)
        x1_u = x1_u + sinusoidal_embedding(t, x1_u.shape[2]).unsqueeze(1)

        # Concat High
        z_f = torch.concat([z1_u, x1_u], dim=1)
        # z_f = z_f + sinusoidal_embedding(t, z_f.shape[2]).unsqueeze(1)

        # B_3ranck Processing
        z_f_3 = self.f1_3(z_f) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f2_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f6_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f7_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f8_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f9_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f3_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_3 = self.f4_3(z_f_3) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        z_f = self.f5_3(z_f_3)

        # z_f_5 = self.f1_5(z_f) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_5 = self.f2_5(z_f_5) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_5 = self.f3_5(z_f_5) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_5 = self.f4_5(z_f_5) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_5 = self.f5_5(z_f_5) + sinusoidal_embedding(t, 1024).unsqueeze(1)

        # z_f_7 = self.f1_7(z_f) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_7 = self.f2_7(z_f_7) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_7 = self.f3_7(z_f_7) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_7 = self.f4_7(z_f_7) + sinusoidal_embedding(t, 1024).unsqueeze(1)
        # z_f_7 = self.f5_7(z_f_7) + sinusoidal_embedding(t, 1024).unsqueeze(1)

        # z_f = torch.concat([z_f_3, z_f_5, z_f_7], dim=1) + sinusoidal_embedding(t, 1024).unsqueeze(1)

        # # Final Processing
        # z_f = self.final(z_f)
        

        # Feed to fully-connected
        # z_flatten = z_f.flatten(1,2)
        # z_f = z_flatten.squeeze()
        # z_f = self.fc(z_f)


        return z_f





