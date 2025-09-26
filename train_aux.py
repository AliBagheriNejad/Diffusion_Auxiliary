# training 
import numpy as np
import os
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import src.utils as utils
import src.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data
X_train_scaled_tensor = utils.read_pkl('Data/1024/features_train.pkl')
X_test_scaled_tensor = utils.read_pkl('Data/1024/features_test.pkl')
y_train_tensor = utils.read_pkl('Data/1024/label_train.pkl')
y_test_tensor = utils.read_pkl('Data/1024/label_test.pkl')

train_loader = utils.make_loader(X_train_scaled_tensor,y_train_tensor, 128)
test_loader = utils.make_loader(X_test_scaled_tensor,y_test_tensor, 128)

# Load models
weight_dir = r'F:\thesis\Articles\2nd\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
weight_dir = os.path.join(weight_dir,'test_weight.pth')
network = utils.load_model(models.Network(26), weight_dir)

weight_dir = r'F:\thesis\Articles\2nd\cod\others\aux_weight_1.pth'
auxiliary =  models.AuxNet(n_layer=1, in_dim=1024*5, out_dim=1024, best_acc=-100)
auxiliary.save_path = 'model_weights_aux.pth'

feature_extractor = network.feature_extractor
classifier = network.classifier



# Training just auxiliary
# ===============================================================
MODE = 'aux_gan'
MODEL = auxiliary
EPOCHS = 10
TRAIN_DATALOADER = train_loader
TEST_DATALOADER = test_loader
OPTIMIZER = optim.Adam(auxiliary.parameters(), lr=0.00001)
CRITERION = nn.CrossEntropyLoss()
EARLY_STOPPING = 'test_loss'
SHOW_GRAD = False

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

train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
for epoch in range(EPOCHS):
    MODEL.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'Epoch {epoch + 1}/{EPOCHS}')

    for i,(batch_data, batch_labels) in progress_bar:
         
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        batch_label = batch_labels[:,2].to(device)
        
        OPTIMIZER.zero_grad()

        batch_data = batch_data.flatten(1,2)
        features = MODEL(batch_data)
        outputs = classifier(features)
        loss = CRITERION(outputs, batch_label)

        loss.backward()
        OPTIMIZER.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += batch_labels.size(0)
        correct_train += (predicted == batch_label).sum().item()

        progress_bar.set_postfix_str(
            f'train_loss={train_loss / (i + 1):.4f}\
            , train_acc={100 * correct_train / total_train:.4f}')
        
    train_loss_log = train_loss / len(TRAIN_DATALOADER)
    train_acc_log = 100 * correct_train / total_train

    train_losses.append(train_loss_log)
    train_accs.append(train_acc_log)

    MODEL.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch_data, batch_labels in TEST_DATALOADER:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_label = batch_labels[:,2]
            
            batch_data = batch_data.flatten(1,2)
            features = MODEL(batch_data)
            outputs = classifier(features)
            loss = CRITERION(outputs, batch_label)


            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_label).sum().item()

            test_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            total_test += batch_labels.size(0)
            correct_test += (predicted == batch_label).sum().item()

    test_loss_log = test_loss / len(TEST_DATALOADER)
    test_acc_log = 100 * correct_test / total_test
    test_losses.append(test_loss_log)
    test_accs.append(test_acc_log)

    print(f'val_loss: {test_losses[-1]:.4f}, val_acc: {test_accs[-1]:.1f}', end='\n')

    MODEL.metrics_now = {
                'train_loss': -train_loss_log,
                'train_acc': train_acc_log,
                'val_acc': test_acc_log,
                'val_loss': -test_loss_log,
            }

    if SHOW_GRAD:
        for name, param in MODEL.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item():.10f}")

    if EARLY_STOPPING == 'test_acc':
        do_break = MODEL.early_stopping(test_accs[-1],epoch)
    elif EARLY_STOPPING == 'test_loss':
        do_break = MODEL.early_stopping(-test_losses[-1],epoch)
    elif EARLY_STOPPING == 'train_acc':
        do_break = MODEL.early_stopping(train_accs[-1],epoch)
    elif EARLY_STOPPING == 'train_loss':
        do_break = MODEL.early_stopping(-train_losses[-1],epoch)
        
    if do_break:
        break
# ===============================================================