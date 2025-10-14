# training 
import numpy as np
import os
import tqdm
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


import src.utils as utils
import src.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data X
data_dir = 'Data'
file_name = 'input.pkl'
file_name_label = 'output.pkl'
file_path = os.path.join(data_dir, file_name)
file_path_label = os.path.join(data_dir, file_name_label)

data = utils.read_pkl(file_path)
label = utils.read_pkl(file_path_label)

label_encoded, label_map = utils.label_encoder(label)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    label_encoded,
    test_size = 0.2,
    random_state = 69,
    shuffle = True,
)

(X_train_scaled, X_test_scaled),scaler_list = utils.scale_it(X_train, X_test)

X_train_scaled_tensor_x, y_train_tensor_x = utils.tensor_it(X_train_scaled,y_train)
# X_test_scaled_tensor_x, y_test_tensor_x = utils.tensor_it(X_test_scaled,y_test)
X_train_scaled_tensor_x, _ = utils.tensor_it(X_train_scaled,y_train)
X_test_scaled_tensor_x, _ = utils.tensor_it(X_test_scaled,y_test)

# Load Data Z
X_train_scaled_tensor = utils.read_pkl('Data/aux_features/features_train.pkl')
X_test_scaled_tensor = utils.read_pkl('Data/aux_features/features_test.pkl')
y_train_tensor = utils.read_pkl('Data/aux_features/labels_train.pkl')
y_test_tensor = utils.read_pkl('Data/aux_features/labels_test.pkl')

# Data Loader
train_loader = utils.make_loader(
    X_train_scaled_tensor,
    X_train_scaled_tensor_x[:,2,:,:],
    y_train_tensor,
    bs = 128
)
test_loader = utils.make_loader(
    X_test_scaled_tensor,
    X_test_scaled_tensor_x[:,2,:,:],
    y_test_tensor,
    bs = 8
)
test_loader_of = utils.make_loader(
    X_test_scaled_tensor[:160,:],
    X_test_scaled_tensor_x[:160,2,:,:],
    y_test_tensor[:160],
    bs = 8
)

model = models.UNET(1,2,2)
model.save_path = 'temp/model_weight.pth'
model.patience = 10

# weight_dir = r'F:\thesis\Articles\2nd\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
# weight_dir = os.path.join(weight_dir,'test_weight.pth')
# network = utils.load_model(models.Network(26), weight_dir)
network = models.Network(26)
cls = network.classifier


# Training UNET (diffuxion model)
# ==================================================
MODE = 'diffusion'
MODEL = model
MODEL_CLS = cls
EPOCHS = 200
TRAIN_DATALOADER = test_loader_of
TEST_DATALOADER = test_loader
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=0.001)
# OPTIMIZER_CLS = optim.Adam(MODEL_CLS.parameters(), lr = 0.00001)
CRITERION = nn.MSELoss()
# CRITERION_CLS = nn.CrossEntropyLoss()
EARLY_STOPPING = 'train_loss'
SHOW_GRAD = True
T = 50

grad_dic = dict()
weight_dic = dict()

MODEL.weight_dic = {
    'train_loss': None,
    'train_gen_loss': None,
    'test_gen_loss': None,
    'test_loss': None
}

MODEL.metrics_best = {
    'train_loss': -100,
    'train_gen_loss': -100,
    'test_gen_loss': -100,
    'test_loss': -100
}

MODEL.best_acc = MODEL.metrics_best[EARLY_STOPPING]

def fix_temp():
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def save_weight_dic():
        for k,v in zip(MODEL.weight_dic.keys(), MODEL.weight_dic.values()):
            weight_name = f'{MODE}_{k}_{np.abs(MODEL.metrics_best[k]):.6f}.pth'
            weight_path = os.path.join('temp', weight_name)
            torch.save(v, weight_path)
            print(f'Weight <{weight_path}> saved successfully')

fix_temp()
dfp = models.DiffusionProcess(T, 0.0002, 0.05)

train_losses, train_gen_losses, test_losses, test_gen_losses = [], [], [], []
train_losses_iter = []

train_cls_losses, train_cls_losses_iter = [], []

for epoch in range(EPOCHS):

    MODEL.train()
    train_loss = 0.0
    train_gen_loss = 0.0
    train_cls_loss = 0.0
    test_loss = 0.0
    test_gen_loss = 0.0

    progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'Epoch {epoch + 1}/{EPOCHS}')

    for i,(batch_z, batch_x, batch_y) in progress_bar:
         
        batch_z = batch_x.to(device).permute(0,2,1)
        batch_x = batch_x.to(device).permute(0,2,1)
        batch_x = torch.zeros_like(batch_x)
        batch_label = batch_y.to(device)

        t = torch.randint(0, T, (batch_z.shape[0],), device=device)

        batch_z_noisy, batch_noise = dfp.q_sample(batch_z,t)
        if (batch_z_noisy.shape) == 2:
            batch_z_noisy = batch_z_noisy.unsqueeze(1)

        OPTIMIZER.zero_grad()

        noise_hat = MODEL(batch_x, batch_z_noisy, t)
        noise_hat = noise_hat.squeeze()

        # batch_z_t = dfp.p_sample(batch_z_noisy, t, noise_hat)
        # output_cls = MODEL_CLS(batch_z_t.detach())

        loss = CRITERION(noise_hat, batch_noise.repeat(1,384//2,1))
        # loss_cls = CRITERION_CLS(output_cls, batch_label)
        # loss = loss_mse + loss_cls
        loss.backward()
        clip_grad_norm_(MODEL.parameters(), max_norm=0.1)
        OPTIMIZER.step()

        # OPTIMIZER_CLS.zero_grad()
        # loss_cls = CRITERION_CLS(MODEL_CLS(batch_z_t.detach()), batch_label)
        # loss_cls.backward()
        # OPTIMIZER_CLS.step()

        loss_mse_np = loss.cpu().detach().numpy()
        # loss_cls_np = loss_cls.cpu().detach().numpy()

        train_losses_iter.append(loss_mse_np)
        train_loss += loss_mse_np
        # train_cls_losses_iter.append(loss_cls_np)
        # train_cls_loss += loss_cls_np
        progress_bar.set_postfix_str(f'train_loss_mse={train_loss / (i + 1):.4f}, train_loss_cls={train_cls_loss / (i + 1):.4f}')

        
    
    train_losses.append(train_loss/len(TRAIN_DATALOADER))
    train_cls_losses.append(train_cls_loss/len(TRAIN_DATALOADER))

  
    if SHOW_GRAD:
        epoch_dic = dict()
        epoch_dic_w = dict()
        for name, param in MODEL.named_parameters():
            if param.grad is not None:
                # print(f"{name}: {param.grad.mean().item():.10f}")
                epoch_dic[f'{name}'] = param.grad.mean().item()
                epoch_dic_w[f'{name}'] = param.mean().item()
        grad_dic[epoch] = epoch_dic
        weight_dic[epoch] = epoch_dic_w


    MODEL.eval()

    with torch.no_grad():
        # progress_bar_sample = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc='\tTrain sampling')
        # for i,(batch_z, batch_x, _) in progress_bar_sample:
            
        #     batch_z_t = torch.randn_like(batch_z)
        #     batch_z_t = batch_z_t.to(device)

        #     batch_z = batch_z.to(device)
        #     batch_x = batch_x.to(device).permute(0,2,1)

        #     for t in range(T-1,-1, -1):

        #         t = torch.full((batch_z.shape[0],), t, device=device)
        #         t_embd = models.sinusoidal_embedding(t, 128).unsqueeze(1)

        #         noise_hat = MODEL(batch_x, batch_z_t.unsqueeze(1), t_embd)
        #         noise_hat = noise_hat.squeeze()

        #         batch_z_t = dfp.p_sample(batch_z_t, t, noise_hat)

        #     loss = CRITERION(batch_z_t, batch_z)
        #     train_gen_loss += loss.cpu().detach().numpy()
        #     progress_bar_sample.set_postfix_str(f'train_gen_loss={train_gen_loss / (i + 1):.4f}')

        train_gen_losses.append(train_gen_loss/len(TRAIN_DATALOADER))

        test_loss = 0.0
        test_gen_loss = 0.0

        # progress_bar_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc=f'\tTest set')

        # for i,(batch_z, batch_x, _) in progress_bar_test:
            
        #     batch_z = batch_z.to(device)
        #     batch_x = batch_x.to(device).permute(0,2,1)

        #     t = torch.randint(0, T, (batch_z.shape[0],), device=device)

        #     batch_z_noisy, batch_noise = dfp.q_sample(batch_z,t)

        #     noise_hat = MODEL(batch_x, batch_z_noisy.unsqueeze(1), t)
        #     noise_hat = noise_hat.squeeze()

        #     loss = CRITERION(noise_hat, batch_noise)

        #     test_loss += loss.cpu().detach().numpy()
        #     progress_bar_test.set_postfix_str(f'test_loss={test_loss / (i + 1):.4f}')
        
        test_losses.append(test_loss/len(TEST_DATALOADER))


        # progress_bar_sample_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc='\tTest sampling')
        # for i,(batch_z, batch_x, _) in progress_bar_sample_test:
            
        #     batch_z_t = torch.randn_like(batch_z)
        #     batch_z_t = batch_z_t.to(device)

        #     batch_z = batch_z.to(device)
        #     batch_x = batch_x.to(device).permute(0,2,1)

        #     for t in range(T-1,-1, -1):

        #         t = torch.full((batch_z.shape[0],), t, device=device)
        #         t_embd = models.sinusoidal_embedding(t, 128).unsqueeze(1)

        #         noise_hat = MODEL(batch_x, batch_z_t.unsqueeze(1), t_embd)
        #         noise_hat = noise_hat.squeeze()

        #         batch_z_t = dfp.p_sample(batch_z_t, t, noise_hat)

        #     loss = CRITERION(batch_z_t, batch_z)
        #     test_gen_loss += loss.cpu().detach().numpy()
        #     progress_bar_sample_test.set_postfix_str(f'test_gen_loss={test_gen_loss / (i + 1):.4f}')

        test_gen_losses.append(test_gen_loss/len(TEST_DATALOADER))


    MODEL.metrics_now = {
                'train_loss': -train_losses[-1],
                'train_gen_loss': -train_gen_losses[-1],
                'test_gen_loss': -test_gen_losses[-1],
                'test_loss': -test_losses[-1],
    } 


    if EARLY_STOPPING == 'test_gen_loss':
        do_break = MODEL.early_stopping(test_gen_losses[-1],epoch)
    elif EARLY_STOPPING == 'test_loss':
        do_break = MODEL.early_stopping(-test_losses[-1],epoch)
    elif EARLY_STOPPING == 'train_gen_loss':
        do_break = MODEL.early_stopping(train_gen_losses[-1],epoch)
    elif EARLY_STOPPING == 'train_loss':
        do_break = MODEL.early_stopping(-train_losses[-1],epoch)
        
    if do_break:
        break


import json
with open("grad.json", "w") as json_file:
    json.dump(grad_dic, json_file, indent=4)
with open("weight.json", "w") as json_file:
    json.dump(weight_dic, json_file, indent=4)


import matplotlib.pyplot as plt
plt.plot(train_losses_iter)
plt.show()



