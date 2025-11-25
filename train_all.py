# training 
import numpy as np
import os
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import mlflow
import inspect     

import src.utils as utils
import src.models as models
from src.models import AuxNet
import src.transformer as transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATIENCE = 10
MODEL_ERATIO = 100
CODE_MODE = 'debug' # 'debug' or 'train
EARLY_STOPPING = 'test_loss'
EPOCHS = 2
LAMBDA = 1
T = 5
GEN_PAI = 20


def load_data(data_dir='Data'):
    # Load data X
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
    X_test_scaled_tensor_x, y_test_tensor_x = utils.tensor_it(X_test_scaled,y_test)

    return X_train_scaled_tensor_x, y_train_tensor_x, X_test_scaled_tensor_x, y_test_tensor_x

def data_loader_model_1(limit=2):
    # Data Loader
    train_loader = utils.make_loader(
        X_train_scaled_tensor[:,limit,:,:],
        y_train_tensor[:,limit],
        bs = 32
    )
    test_loader = utils.make_loader(
        X_test_scaled_tensor[:,limit,:,:],
        y_test_tensor[:,limit],
        bs = 8
    )
    test_loader_of = utils.make_loader(
        X_test_scaled_tensor[:160,limit,:,:],
        y_test_tensor[:160,limit],
        bs = 8
    )
    return train_loader, test_loader, test_loader_of

def data_loader_model_2():
        # Data Loader
    train_loader = utils.make_loader(
        X_train_scaled_tensor_z1,
        y_train_tensor[:,2],
        bs = 32
    )
    test_loader = utils.make_loader(
        X_test_scaled_tensor_z1,
        y_test_tensor[:,2],
        bs = 8
    )
    test_loader_of = utils.make_loader(
        X_test_scaled_tensor_z1[:160,:,:],
        y_test_tensor[:160,2],
        bs = 8
    )
    return train_loader, test_loader, test_loader_of

def data_loader_model_ft():
    # Data Loader
    train_loader = utils.make_loader(
        X_train_scaled_tensor[:,2,:,:],
        X_train_scaled_tensor_z2,
        y_train_tensor[:,2],
        bs = 32
    )
    test_loader = utils.make_loader(
        X_test_scaled_tensor[:,2,:,:],
        X_test_scaled_tensor_z2,
        y_test_tensor[:,2],
        bs = 8
    )
    test_loader_of = utils.make_loader(
        X_test_scaled_tensor[:160,2,:,:],
        X_test_scaled_tensor_z2[:160,:],
        y_test_tensor[:160,2],
        bs = 8
    )
    return train_loader, test_loader, test_loader_of

def data_loader_model_r():
    # Data Loader
    train_loader = utils.make_loader(
        X_train_scaled_tensor[:,2,:,:],
        X_train_scaled_tensor_z2,
        X_train_scaled_tensor_d,
        y_train_tensor[:,2],
        bs = 32
    )
    test_loader = utils.make_loader(
        X_test_scaled_tensor[:,2,:,:],
        X_test_scaled_tensor_z2,
        X_test_scaled_tensor_d,
        y_test_tensor[:,2],
        bs = 8
    )
    test_loader_of = utils.make_loader(
        X_test_scaled_tensor[:160,2,:,:],
        X_test_scaled_tensor_z2[:160,:],
        X_test_scaled_tensor_d[:160,:],
        y_test_tensor[:160,2],
        bs = 8
    )
    return train_loader, test_loader, test_loader_of

def fix_temp():
    '''
    Giving out a clean temp folder
    '''

    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


X_train_scaled_tensor, y_train_tensor, X_test_scaled_tensor, y_test_tensor = load_data()

train_loader, test_loader, test_loader_of = data_loader_model_1()
train_loader2, test_loader2, test_loader_of2 = data_loader_model_1([0,1,2,3,4])

EX_NAME = 'CWRU'
# =====================================================================================================
# MODEL 1
description = ''
# model initialization
model_params = {
    'num_classes':26,
    'in_channels':2,
}
model = models.Network(**model_params).to(device)
model.save_path = f'temp/model_1_{EX_NAME}.pth'
model.patience = MODEL_PATIENCE
model.e_ratio = MODEL_ERATIO
model.weight_dic = {
    'train_loss': None,
    'train_acc': None,
    'test_acc': None,
    'test_loss': None
}
model.metrics_best = {
    'train_loss': -100,
    'train_acc': -100,
    'test_acc': -100,
    'test_loss': -100
}
model.best_acc = model.metrics_best[EARLY_STOPPING]
run_name = 'MODEL_1'

# training data
if CODE_MODE == 'debug':
    TRAIN_DATALOADER = test_loader_of
    TEST_DATALOADER = test_loader_of
    TRAIN_DATALOADER2= train_loader2
    TEST_DATALOADER2 = test_loader2
elif CODE_MODE == 'train':
    TRAIN_DATALOADER = train_loader
    TEST_DATALOADER = test_loader
    TRAIN_DATALOADER2 = train_loader2
    TEST_DATALOADER2 = test_loader2

# training tools
OPTIMIZER = optim.Adam(model.parameters(), lr=0.001)
CRITERION = nn.CrossEntropyLoss()

# MLFlow initialization
fix_temp()
mlflow.set_experiment(EX_NAME)
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name=run_name) as run:
    run_id_model_1 = run.info.run_id
    mlflow.set_tag('dscr',description)
    mlflow.log_params(model_params)

    # Create a temporary file to store the source code
    source_code = inspect.getsource(models.Network)
    temp_code_file = f"temp/model_1_{EX_NAME}_definition.txt"
    with open(temp_code_file, "w") as f:
        f.write(source_code)

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    # trainig loop
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        print()
        progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'{EX_NAME}_{run_name}_EPOCH {epoch + 1}/{EPOCHS}')

        for i,(batch_x, batch_y) in progress_bar:

            batch_x = batch_x.to(device)
            batch_label = batch_y.to(device)        
            
            OPTIMIZER.zero_grad()

            outputs = model(batch_x)

            loss = CRITERION(outputs, batch_label)
            loss.backward()

            OPTIMIZER.step()


            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_label.size(0)
            correct_train += (predicted == batch_label).sum().item()

            progress_bar.set_postfix_str(
                f'train_loss={train_loss / (i + 1):.4f}\
                , train_acc={100 * correct_train / total_train:.4f}')

            
        
        train_losses.append(train_loss/len(TRAIN_DATALOADER))
        train_accs.append(100 * correct_train / total_train)


        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():

            progress_bar_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc=f'test set')

            for i,(batch_x, batch_y) in progress_bar_test:

                batch_x = batch_x.to(device)
                batch_label = batch_y.to(device)        
                
                outputs = model(batch_x)

                loss = CRITERION(outputs, batch_label)

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += batch_label.size(0)
                correct_train += (predicted == batch_label).sum().item()

                progress_bar_test.set_postfix_str(
                    f'test_loss={test_loss / (i + 1):.4f}\
                    , test_acc={100 * correct_test / total_test:.4f}')
                
            test_accs.append(100 * correct_test / total_test)
            test_losses.append(test_loss/len(TEST_DATALOADER))


        model.metrics_now = {
                    'train_loss': -train_losses[-1],
                    'train_acc': train_accs[-1],
                    'test_acc': test_accs[-1],
                    'test_loss': -test_losses[-1],
        } 

        mlflow.log_metric('train_loss', train_losses[-1], step=epoch)
        mlflow.log_metric('test_loss', test_losses[-1], step=epoch) 
        mlflow.log_metric('train_acc', train_accs[-1], step=epoch)
        mlflow.log_metric('test_acc', test_accs[-1], step=epoch) 

        if EARLY_STOPPING == 'test_acc':
            do_break = model.early_stopping(test_accs[-1],epoch)
        elif EARLY_STOPPING == 'test_loss':
            do_break = model.early_stopping(-test_losses[-1],epoch)
        elif EARLY_STOPPING == 'train_acc':
            do_break = model.early_stopping(train_accs[-1],epoch)
        elif EARLY_STOPPING == 'train_loss':
            do_break = model.early_stopping(-train_losses[-1],epoch)

        if do_break:
            break
        mlflow.log_artifacts('temp', 'artifacts')

    # Save features
    model = torch.load(model.save_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        train_list = []
        test_list = []
        for batch_data, _ in TRAIN_DATALOADER2:
            x = batch_data.to(device)
            x = x.permute(0,1,3,2)
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
            features = model.feature_extractor(x) 
            features = features.view(batch_data.shape[0], batch_data.shape[1],-1)           
            train_list.append(features)

        for batch_data, _ in TEST_DATALOADER2:
            x = batch_data.to(device)
            x = x.permute(0,1,3,2)
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
            features = model.feature_extractor(x) 
            features = features.view(batch_data.shape[0], batch_data.shape[1],-1)           
            test_list.append(features)

    train_features = torch.concat(train_list, dim=0)
    test_features = torch.concat(test_list, dim=0)
    with open('temp/features_train_model1.pkl', 'wb') as file:
        pickle.dump(train_features, file)
    with open('temp/features_test_model1.pkl', 'wb') as file:
        pickle.dump(test_features, file)
    
    mlflow.log_artifacts('temp', 'artifacts')

# =====================================================================================================

# run_id_model_1 = '9c0eef5e60c74e1fb68a9e5847e47cf7'
# Load model_1
client_model1 = mlflow.tracking.MlflowClient()
local_path_model1 = client_model1.download_artifacts(run_id_model_1, 'artifacts/'+f'model_1_{EX_NAME}.pth')
model_model_1 = torch.load(local_path_model1, weights_only=False)
classifier_model_1 = model_model_1.classifier

# Load model_1 features 
feature_model1_train_path = client_model1.download_artifacts(run_id_model_1, 'artifacts/features_train_model1.pkl')
feature_model1_test_path = client_model1.download_artifacts(run_id_model_1, 'artifacts/features_test_model1.pkl')

X_train_scaled_tensor_z1 = utils.read_pkl(feature_model1_train_path)
X_test_scaled_tensor_z1 = utils.read_pkl(feature_model1_test_path)

train_loader, test_loader, test_loader_of = data_loader_model_2()

# MODEL 2
description = ''
# model initialization
model_params = {
    'n_layer':1,
    'in_dim':1024*5,
    'out_dim':1024,
    'best_acc':-100
}
model = models.AuxNet(**model_params)
model.save_path = f'temp/model_2_{EX_NAME}.pth'
model.patience = MODEL_PATIENCE
model.e_ratio = MODEL_ERATIO
model.weight_dic = {
    'train_loss': None,
    'train_acc': None,
    'test_acc': None,
    'test_loss': None
}
model.metrics_best = {
    'train_loss': -100,
    'train_acc': -100,
    'test_acc': -100,
    'test_loss': -100
}
model.best_acc = model.metrics_best[EARLY_STOPPING]
run_name = 'MODEL_2'

# training data
if CODE_MODE == 'debug':
    TRAIN_DATALOADER = test_loader_of
    TEST_DATALOADER = test_loader_of
elif CODE_MODE == 'train':
    TRAIN_DATALOADER = train_loader
    TEST_DATALOADER = test_loader

# training tools
OPTIMIZER = optim.Adam(model.parameters(), lr=0.001)
CRITERION = nn.CrossEntropyLoss()

# MLFlow initialization
fix_temp()
mlflow.set_experiment(EX_NAME)
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name=run_name) as run:
    run_id_model_2 = run.info.run_id
    mlflow.set_tag('dscr',description)
    mlflow.log_params(model_params)

    # Create a temporary file to store the source code
    source_code = inspect.getsource(models.AuxNet)
    temp_code_file = f"temp/model_2_{EX_NAME}_definition.txt"
    with open(temp_code_file, "w") as f:
        f.write(source_code)

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    # trainig loop
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        print()
        progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'{EX_NAME}_{run_name}_EPOCH {epoch + 1}/{EPOCHS}')

        for i,(batch_z, batch_y) in progress_bar:

            batch_z = batch_z.to(device)
            batch_label = batch_y.to(device)        
            
            OPTIMIZER.zero_grad()

            batch_z = batch_z.flatten(1,2)
            features = model(batch_z)
            outputs = classifier_model_1(features)

            loss = CRITERION(outputs, batch_label)
            loss.backward()

            OPTIMIZER.step()


            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_label.size(0)
            correct_train += (predicted == batch_label).sum().item()

            progress_bar.set_postfix_str(
                f'train_loss={train_loss / (i + 1):.4f}\
                , train_acc={100 * correct_train / total_train:.4f}')

            
        
        train_losses.append(train_loss/len(TRAIN_DATALOADER))
        train_accs.append(100 * correct_train / total_train)


        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():

            progress_bar_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc=f'test set')

            for i,(batch_z, batch_y) in progress_bar_test:

                batch_z = batch_z.to(device)
                batch_label = batch_y.to(device)    

                batch_z = batch_z.flatten(1,2)
                features = model(batch_z)
                outputs = classifier_model_1(features)

                loss = CRITERION(outputs, batch_label)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += batch_label.size(0)
                correct_test += (predicted == batch_label).sum().item()

                progress_bar_test.set_postfix_str(
                    f'trest_loss={test_loss / (i + 1):.4f}\
                    , test_acc={100 * correct_test / total_test:.4f}')
                
            test_accs.append(100 * correct_test / total_test)
            test_losses.append(test_loss/len(TEST_DATALOADER))


        model.metrics_now = {
                    'train_loss': -train_losses[-1],
                    'train_acc': train_accs[-1],
                    'test_acc': test_accs[-1],
                    'test_loss': -test_losses[-1],
        } 

        mlflow.log_metric('train_loss', train_losses[-1], step=epoch)
        mlflow.log_metric('test_loss', test_losses[-1], step=epoch) 
        mlflow.log_metric('train_acc', train_accs[-1], step=epoch)
        mlflow.log_metric('test_acc', test_accs[-1], step=epoch) 

        if EARLY_STOPPING == 'test_acc':
            do_break = model.early_stopping(test_accs[-1],epoch)
        elif EARLY_STOPPING == 'test_loss':
            do_break = model.early_stopping(-test_losses[-1],epoch)
        elif EARLY_STOPPING == 'train_acc':
            do_break = model.early_stopping(train_accs[-1],epoch)
        elif EARLY_STOPPING == 'train_loss':
            do_break = model.early_stopping(-train_losses[-1],epoch)

        if do_break:
            break
        mlflow.log_artifacts('temp', 'artifacts')

    # Save features
    model = torch.load(model.save_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        train_list = []
        test_list = []
        for batch_data, _ in train_loader:
            x = batch_data.to(device)
            batch_data = batch_data.flatten(1,2)
            features = model(batch_data)          
            train_list.append(features)

        for batch_data, _ in test_loader:
            x = batch_data.to(device)
            batch_data = batch_data.flatten(1,2)
            features = model(batch_data)           
            test_list.append(features)

    train_features = torch.concat(train_list, dim=0)
    test_features = torch.concat(test_list, dim=0)
    with open('temp/features_train_model2.pkl', 'wb') as file:
        pickle.dump(train_features, file)
    with open('temp/features_test_model2.pkl', 'wb') as file:
        pickle.dump(test_features, file)
    
    mlflow.log_artifacts('temp', 'artifacts')

# =====================================================================================================

# run_id_model_2 = '0df6a72c360248c5b0a2416e4f40944c'

# Load model_2 features 
client_model2 = mlflow.tracking.MlflowClient()
feature_model2_train_path = client_model2.download_artifacts(run_id_model_2, 'artifacts/features_train_model2.pkl')
feature_model2_test_path = client_model2.download_artifacts(run_id_model_2, 'artifacts/features_test_model2.pkl')

X_train_scaled_tensor_z2 = utils.read_pkl(feature_model2_train_path)
X_test_scaled_tensor_z2 = utils.read_pkl(feature_model2_test_path)

train_loader, test_loader, test_loader_of = data_loader_model_ft()

# MODEL FT
description = ''
# model initialization
model_params = {
    'in_channel_z': 2,
    'out_channel_z': 1
} 
model = models.UNETAE(**model_params)
model.save_path = f'temp/model_ft_{EX_NAME}.pth'
model.patience = MODEL_PATIENCE
model.e_ratio = MODEL_ERATIO
model.weight_dic = {
    'train_loss_mse': None,
    'train_loss_ce': None,
    'train_acc': None,
    'test_acc': None,
    'test_loss_mse': None,
    'test_loss_ce': None
}
model.metrics_best = {
    'train_loss_mse': -100,
    'train_loss_ce': -100,
    'train_acc': -100,
    'test_acc': -100,
    'test_loss_mse': -100,
    'test_loss_ce': -100
}
EARLY_STOPPING2 = EARLY_STOPPING+'_mse'
model.best_acc = model.metrics_best[EARLY_STOPPING2]
run_name = 'MODEL_FT'

# training data
if CODE_MODE == 'debug':
    TRAIN_DATALOADER = test_loader_of
    TEST_DATALOADER = test_loader_of
elif CODE_MODE == 'train':
    TRAIN_DATALOADER = train_loader
    TEST_DATALOADER = test_loader

# training tools
OPTIMIZER = optim.Adam(model.parameters(), lr=0.001)
CRITERION_CLS = nn.CrossEntropyLoss()
CRITERION_MSE = nn.MSELoss()

# MLFlow initialization
fix_temp()
mlflow.set_experiment(EX_NAME)
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name=run_name) as run:
    run_id_model_ft = run.info.run_id
    mlflow.set_tag('dscr',description)
    mlflow.log_params(model_params)

    # Create a temporary file to store the source code
    source_code = inspect.getsource(models.UNETAE)
    temp_code_file = f"temp/model_ft_{EX_NAME}_definition.txt"
    with open(temp_code_file, "w") as f:
        f.write(source_code)

    train_losses_mse, train_losses_ce, train_accs, test_losses_mse, test_losses_ce, test_accs = [], [], [], [], [], []

    # trainig loop
    for epoch in range(EPOCHS):

        model.train()
        train_loss_mse = 0.0
        train_loss_ce = 0.0
        correct_train = 0
        total_train = 0
        print()
        progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'{EX_NAME}_{run_name}_EPOCH {epoch + 1}/{EPOCHS}')

        for i,(batch_x, batch_z, batch_y) in progress_bar:

            batch_x = batch_x.to(device).permute(0,2,1)
            batch_z = batch_z.to(device)
            batch_label = batch_y.to(device)        
            
            OPTIMIZER.zero_grad()

            features = model(batch_x)
            outputs = classifier_model_1(features).squeeze()

            loss_cls = CRITERION_CLS(outputs, batch_label) * LAMBDA
            loss_mse = CRITERION_MSE(features.squeeze(), batch_z) 
            loss = loss_cls + loss_mse
            loss.backward()

            OPTIMIZER.step()


            train_loss_mse += loss_mse.item()
            train_loss_ce += loss_cls.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_label.size(0)
            correct_train += (predicted == batch_label).sum().item()

            progress_bar.set_postfix_str(
                f'train_loss_mse={train_loss_mse / (i + 1):.4f}, train_loss_ce={train_loss_ce / (i + 1):.4f} \
, train_acc={100 * correct_train / total_train:.4f}')

            
        
        train_losses_mse.append(train_loss_mse/len(TRAIN_DATALOADER))
        train_losses_ce.append(train_loss_ce/len(TRAIN_DATALOADER))
        train_accs.append(100 * correct_train / total_train)


        model.eval()
        test_loss_mse = 0.0
        test_loss_ce = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():

            progress_bar_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc=f'test set')

            for i,(batch_x, batch_z, batch_y) in progress_bar_test:

                batch_x = batch_x.to(device).permute(0,2,1)
                batch_z = batch_z.to(device)
                batch_label = batch_y.to(device) 

                features = model(batch_x)
                outputs = classifier_model_1(features).squeeze()

                loss_cls = CRITERION_CLS(outputs, batch_label) * LAMBDA
                loss_mse = CRITERION_MSE(features.squeeze(), batch_z)

                test_loss_mse += loss_mse.item()
                test_loss_ce += loss_cls.item()
                _, predicted = torch.max(outputs, 1)
                total_test += batch_label.size(0)
                correct_test += (predicted == batch_label).sum().item()

                progress_bar_test.set_postfix_str(
                    f'test_loss_mse={test_loss_mse / (i + 1):.4f}, test_loss_ce={test_loss_ce / (i + 1):.4f} \
, test_acc={100 * correct_test / total_test:.4f}')

                        
                    
            test_losses_mse.append(test_loss_mse/len(TEST_DATALOADER))
            test_losses_ce.append(test_loss_ce/len(TEST_DATALOADER))
            test_accs.append(100 * correct_test / total_test)


        model.metrics_now = {
                    'train_loss_mse': -train_losses_mse[-1],
                    'train_loss_ce': -train_losses_ce[-1],
                    'train_acc': train_accs[-1],
                    'test_acc': test_accs[-1],
                    'test_loss_mse': -test_losses_mse[-1],
                    'test_loss_ce': -test_losses_ce[-1],
        } 

        mlflow.log_metric('train_loss_mse', train_losses_mse[-1], step=epoch)
        mlflow.log_metric('test_loss_mse', test_losses_mse[-1], step=epoch) 
        mlflow.log_metric('train_loss_ce', train_losses_ce[-1], step=epoch)
        mlflow.log_metric('test_loss_ce', test_losses_ce[-1], step=epoch) 
        mlflow.log_metric('train_acc', train_accs[-1], step=epoch)
        mlflow.log_metric('test_acc', test_accs[-1], step=epoch) 

        if EARLY_STOPPING2 == 'test_acc':
            do_break = model.early_stopping(test_accs[-1],epoch)
        elif EARLY_STOPPING2 == 'test_loss_mse':
            do_break = model.early_stopping(-test_losses_mse[-1],epoch)
        elif EARLY_STOPPING2 == 'test_loss_ce':
            do_break = model.early_stopping(-test_losses_ce[-1],epoch)
        elif EARLY_STOPPING2 == 'train_acc':
            do_break = model.early_stopping(train_accs[-1],epoch)
        elif EARLY_STOPPING2 == 'train_loss_mse':
            do_break = model.early_stopping(-train_losses_mse[-1],epoch)
        elif EARLY_STOPPING2 == 'train_loss_ce':
            do_break = model.early_stopping(-train_losses_ce[-1],epoch)

        if do_break:
            break
        mlflow.log_artifacts('temp', 'artifacts')

    # Save confusion matrix
    model = torch.load(model.save_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        train_list = []
        test_list = []
        train_list_y = []
        test_list_y = []
        for batch_data, _, batch_label in train_loader:
            label = batch_label.to(device)
            batch_data = batch_data.to(device).permute(0,2,1)
            features = model(batch_data)
            outputs = classifier_model_1(features) 
            _, outputs_cls = torch.max(outputs.squeeze(1),1) 
            train_list_y.append(label)
            train_list.append(outputs_cls)

        for batch_data, _, batch_label in test_loader:
            label = batch_label.to(device)
            batch_data = batch_data.to(device).permute(0,2,1)
            features = model(batch_data)
            outputs = classifier_model_1(features) 
            _, outputs_cls = torch.max(outputs.squeeze(1),1) 
            test_list_y.append(label)
            test_list.append(outputs_cls)

    train_cls = torch.concat(train_list, dim=0)
    test_cls = torch.concat(test_list, dim=0)
    train_cls_y = torch.concat(train_list_y, dim=0)
    test_cls_y = torch.concat(test_list_y, dim=0)

    cm_train = confusion_matrix(train_cls_y.cpu(), train_cls.cpu())
    cm_test = confusion_matrix(test_cls_y.cpu(), test_cls.cpu())
    report_train = classification_report(train_cls_y.cpu(), train_cls.cpu())
    report_test = classification_report(test_cls_y.cpu(), test_cls.cpu())

    with open('temp/cm_train_FT.pkl', 'wb') as file:
        pickle.dump(cm_train, file)
    with open('temp/cm_test_FT.pkl', 'wb') as file:
        pickle.dump(cm_test, file)
    
    with open('temp/report_train_FT.txt', 'w') as file:
        file.write(report_train)
    with open('temp/report_test_FT.txt', 'w') as file:
        file.write(report_test)
    
    mlflow.log_artifacts('temp', 'artifacts')

# =====================================================================================================

# run_id_model_2 = '0df6a72c360248c5b0a2416e4f40944c'

# Load model_2 features 
client_model2 = mlflow.tracking.MlflowClient()
feature_model2_train_path = client_model2.download_artifacts(run_id_model_2, 'artifacts/features_train_model2.pkl')
feature_model2_test_path = client_model2.download_artifacts(run_id_model_2, 'artifacts/features_test_model2.pkl')

X_train_scaled_tensor_z2 = utils.read_pkl(feature_model2_train_path)
X_test_scaled_tensor_z2 = utils.read_pkl(feature_model2_test_path)

train_loader, test_loader, test_loader_of = data_loader_model_ft()

# MODEL D
description = ''
# model initialization
model_params = {
    'in_channel_z': 1,
    'out_channel_z': 1,
    'in_channel_x': 2
} 
model = models.UNET(**model_params).to(device)
model.save_path = f'temp/model_d_{EX_NAME}.pth'
model.patience = MODEL_PATIENCE
model.e_ratio = MODEL_ERATIO
model.weight_dic = {
    'train_loss': None,
    'train_gen_loss': None,
    'test_gen_loss': None,
    'test_loss': None
}
model.metrics_best = {
    'train_loss': -100,
    'train_gen_loss': -100,
    'test_gen_loss': -100,
    'test_loss': -100
}
model.best_acc = model.metrics_best[EARLY_STOPPING]
run_name = 'MODEL_D'

# training data
if CODE_MODE == 'debug':
    TRAIN_DATALOADER = test_loader_of
    TEST_DATALOADER = test_loader_of
elif CODE_MODE == 'train':
    TRAIN_DATALOADER = train_loader
    TEST_DATALOADER = test_loader

# training tools
OPTIMIZER = optim.Adam(model.parameters(), lr=0.001)
CRITERION = nn.MSELoss()

# Diffusion parameters
dfp_params = {
    'T': T,
    'beta_start': -6,
    'beta_end': -1,
    'beta_type': 'lin'
}
dfp = models.DiffusionProcess(**dfp_params)

# MLFlow initialization
fix_temp()
mlflow.set_experiment(EX_NAME)
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name=run_name) as run:
    run_id_model_d = run.info.run_id
    mlflow.set_tag('dscr',description)
    mlflow.log_params(model_params)
    mlflow.log_params(dfp_params)
    mlflow.set_tag('Time steps', T)

    # Create a temporary file to store the source code
    source_code = inspect.getsource(models.UNET)
    temp_code_file = f"temp/model_ft_{EX_NAME}_definition.txt"
    with open(temp_code_file, "w") as f:
        f.write(source_code)

    train_losses, train_gen_losses, test_losses, test_gen_losses = [], [], [], []

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        train_gen_loss = 0.0
        train_cls_loss = 0.0
        test_loss = 0.0
        test_gen_loss = 0.0
        print()
        progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'{EX_NAME}_{run_name}_EPOCH {epoch + 1}/{EPOCHS}')

        for i,(batch_x, batch_z, _) in progress_bar:
            
            batch_z = batch_z.to(device).unsqueeze(1)
            batch_x = batch_x.to(device).permute(0,2,1)
            OPTIMIZER.zero_grad()

            t = torch.randint(0, T, (batch_z.shape[0],), device=device)

            batch_z_noisy, batch_noise = dfp.q_sample(batch_z,t)

            noise_hat = model(batch_z_noisy, batch_x, t)
            loss = CRITERION(noise_hat, batch_noise)
            loss.backward() 
            OPTIMIZER.step()

            loss_mse_np = loss.cpu().detach().numpy()
            train_loss += loss_mse_np
            progress_bar.set_postfix_str(f'train_loss_mse={train_loss / (i + 1):.4f}')
        
        train_losses.append(train_loss/len(TRAIN_DATALOADER))

        model.eval()
        gen_cond = ((epoch+1)%GEN_PAI==0)
        with torch.no_grad():

            test_loss = 0.0
            test_gen_loss = 0.0
            train_gen_loss = 0.0

            if gen_cond:
                progress_bar_sample = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc='train sampling')
                for i,(batch_x, batch_z, _) in progress_bar_sample:
                    
                    batch_z = batch_z.to(device).unsqueeze(1)
                    batch_x = batch_x.to(device).permute(0,2,1)

                    batch_z_t = torch.randn_like(batch_z)
                    batch_z_t = batch_z_t.to(device)

                    for t in range(T-1,-1, -1):

                        step_t = int(t)
                        
                        t = torch.full((batch_z.shape[0],), t, device=device)

                        noise_hat = model(batch_z_t, batch_x, t)
                        batch_z_t_1 = batch_z_t
                        batch_z_t = dfp.p_sample(batch_z_t, t, noise_hat)
                        
                        step_loss = CRITERION(batch_z_t, batch_z_t_1)
                        # mlflow.log_metric(f'gen_step_loss_{i}', float(step_loss.cpu().detach().numpy()), step=step_t)
                        step_loss = CRITERION(batch_z_t, batch_z)
                        # mlflow.log_metric(f'gen_step_loss_{i}_total', float(step_loss.cpu().detach().numpy()), step=step_t)
                
                    loss = CRITERION(batch_z_t, batch_z)
                    train_gen_loss += loss.cpu().detach().numpy()
                    progress_bar_sample.set_postfix_str(f'train_gen_loss={train_gen_loss / (i + 1):.4f}')

                    if (i > 2) and (CODE_MODE=='debug'):
                        break

                train_gen_losses.append(train_gen_loss/len(TRAIN_DATALOADER))

            progress_bar_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc=f'test set')

            for i,(batch_x, batch_z, _) in progress_bar_test:
                
                batch_z = batch_z.to(device).unsqueeze(1)
                batch_x = batch_x.to(device).permute(0,2,1)

                t = torch.randint(0, T, (batch_z.shape[0],), device=device)

                batch_z_noisy, batch_noise = dfp.q_sample(batch_z,t)

                noise_hat = model(batch_z_noisy, batch_x, t)

                loss = CRITERION(noise_hat, batch_noise)

                test_loss += loss.cpu().detach().numpy()
                progress_bar_test.set_postfix_str(f'test_loss={test_loss / (i + 1):.4f}')
            
            test_losses.append(test_loss/len(TEST_DATALOADER))

            if gen_cond:
                progress_bar_sample_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc='test sampling')
                for i,(batch_x, batch_z, _) in progress_bar_sample_test:
                    
                    batch_z = batch_z.to(device).unsqueeze(1)
                    batch_x = batch_x.to(device).permute(0,2,1)

                    batch_z_t = torch.randn_like(batch_z)
                    batch_z_t = batch_z_t.to(device)

                    for t in range(T-1,-1, -1):

                        t = torch.full((batch_z.shape[0],), t, device=device)

                        noise_hat = model(batch_z_t, batch_x, t)

                        batch_z_t = dfp.p_sample(batch_z_t, t, noise_hat)

                    loss = CRITERION(batch_z_t, batch_z)
                    # torch.save(batch_z_t, 'temp/batch_z_t_test.pt')
                    # torch.save(batch_z, 'temp/batch_z_test.pt')
                    test_gen_loss += loss.cpu().detach().numpy()
                    progress_bar_sample_test.set_postfix_str(f'test_gen_loss={test_gen_loss / (i + 1):.4f}')

                    if (i >2) and (CODE_MODE=='debug') :
                        break 

                test_gen_losses.append(test_gen_loss/len(TEST_DATALOADER))
        model.metrics_now = {
                    'train_loss': -train_losses[-1],
                    'train_gen_loss': -10000,
                    'test_gen_loss': -10000,
                    'test_loss': -test_losses[-1],
        } 

        mlflow.log_metric('train_loss', train_losses[-1], step=epoch)
        mlflow.log_metric('test_loss', test_losses[-1], step=epoch)

        if gen_cond:  
            mlflow.log_metric('train_gen_loss', train_gen_losses[-1], step=epoch)
            mlflow.log_metric('test_gen_loss', test_gen_losses[-1], step=epoch)   

        if EARLY_STOPPING == 'test_gen_loss':
            do_break = model.early_stopping(-test_gen_losses[-1],epoch)
        elif EARLY_STOPPING == 'test_loss':
            do_break = model.early_stopping(-test_losses[-1],epoch)
        elif EARLY_STOPPING == 'train_gen_loss':
            do_break = model.early_stopping(-train_gen_losses[-1],epoch)
        elif EARLY_STOPPING == 'train_loss':
            do_break = model.early_stopping(-train_losses[-1],epoch)

        if do_break:
            break
        mlflow.log_artifacts('temp', 'artifacts') 

    # Save generated features
    print('\nGenerating samples for feature vector')
    model = torch.load(model.save_path, weights_only=False)
    model.eval()
    with torch.no_grad():  

        train_gen_list = []
        test_gen_list = []
        progress_bar_sample = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc='train sampling')
        for i,(batch_x, batch_z, _) in progress_bar_sample:
            
            batch_z = batch_z.to(device).unsqueeze(1)
            batch_x = batch_x.to(device).permute(0,2,1)

            batch_z_t = torch.randn_like(batch_z)
            batch_z_t = batch_z_t.to(device)

            for t in range(T-1,-1, -1):

                step_t = int(t)

                t = torch.full((batch_z.shape[0],), t, device=device)

                noise_hat = model(batch_z_t, batch_x, t)
                batch_z_t_1 = batch_z_t
                batch_z_t = dfp.p_sample(batch_z_t, t, noise_hat)
            
            train_gen_list.append(batch_z_t)

        progress_bar_sample_test = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='test sampling')
        for i,(batch_x, batch_z, _) in progress_bar_sample_test:
            
            batch_z = batch_z.to(device).unsqueeze(1)
            batch_x = batch_x.to(device).permute(0,2,1)

            batch_z_t = torch.randn_like(batch_z)
            batch_z_t = batch_z_t.to(device)

            for t in range(T-1,-1, -1):

                t = torch.full((batch_z.shape[0],), t, device=device)

                noise_hat = model(batch_z_t, batch_x, t)

                batch_z_t = dfp.p_sample(batch_z_t, t, noise_hat)

            test_gen_list.append(batch_z_t)

        train_gen_tensor = torch.concat(train_gen_list, dim=0)
        test_gen_tensor = torch.concat(test_gen_list, dim=0)
        with open('temp/features_train_d.pkl', 'wb') as file:
            pickle.dump(train_gen_tensor, file)
        with open('temp/features_test_d.pkl', 'wb') as file:
            pickle.dump(test_gen_tensor, file)

        mlflow.log_artifacts('temp', 'artifacts')

# =====================================================================================================

# run_id_model_d = 'd7e7ed1887204262922777d728c93ade'

# Load model_d features
client_modeld = mlflow.tracking.MlflowClient()
feature_modeld_train_path = client_modeld.download_artifacts(run_id_model_d, 'artifacts/features_train_d.pkl')
feature_modeld_test_path = client_modeld.download_artifacts(run_id_model_d, 'artifacts/features_test_d.pkl')

X_train_scaled_tensor_d = utils.read_pkl(feature_modeld_train_path)
X_test_scaled_tensor_d = utils.read_pkl(feature_modeld_test_path)

train_loader, test_loader, test_loader_of = data_loader_model_r()

# MODEL D
description = ''
# model initialization
model_params = {
    'in_channel_z': 1,
    'out_channel_z': 1,
    'in_channel_x': 2
} 
model = models.UNET(**model_params).to(device)
model.save_path = f'temp/model_r_{EX_NAME}.pth'
model.patience = MODEL_PATIENCE
model.e_ratio = MODEL_ERATIO
model.weight_dic = {
    'train_loss_mse': None,
    'train_loss_ce': None,
    'train_acc': None,
    'test_acc': None,
    'test_loss_mse': None,
    'test_loss_ce': None
}
model.metrics_best = {
    'train_loss_mse': -100,
    'train_loss_ce': -100,
    'train_acc': -100,
    'test_acc': -100,
    'test_loss_mse': -100,
    'test_loss_ce': -100
}
EARLY_STOPPING2 = EARLY_STOPPING+'_mse'
model.best_acc = model.metrics_best[EARLY_STOPPING2]
run_name = 'MODEL_R'

# training data
if CODE_MODE == 'debug':
    TRAIN_DATALOADER = test_loader_of
    TEST_DATALOADER = test_loader_of
elif CODE_MODE == 'train':
    TRAIN_DATALOADER = train_loader
    TEST_DATALOADER = test_loader

# training tools
OPTIMIZER = optim.Adam(model.parameters(), lr=0.001)
CRITERION_CLS = nn.CrossEntropyLoss()
CRITERION_MSE = nn.MSELoss()

# MLFlow initialization
fix_temp()
mlflow.set_experiment(EX_NAME)
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name=run_name) as run:
    run_id_model_r = run.info.run_id
    mlflow.set_tag('dscr',description)
    mlflow.log_params(model_params)

    # Create a temporary file to store the source code
    source_code = inspect.getsource(models.UNET)
    temp_code_file = f"temp/model_r_{EX_NAME}_definition.txt"
    with open(temp_code_file, "w") as f:
        f.write(source_code)

    train_losses_mse, train_losses_ce, train_accs, test_losses_mse, test_losses_ce, test_accs = [], [], [], [], [], []

    # trainig loop
    for epoch in range(EPOCHS):

        model.train()
        train_loss_mse = 0.0
        train_loss_ce = 0.0
        correct_train = 0
        total_train = 0
        print()
        progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'{EX_NAME}_{run_name}_EPOCH {epoch + 1}/{EPOCHS}')

        for i,(batch_x, batch_z, batch_g, batch_y) in progress_bar:
            
            batch_z = batch_z.to(device)
            batch_g = batch_g.to(device)
            batch_x = batch_x.to(device).permute(0,2,1)
            batch_label = batch_y.to(device)
            OPTIMIZER.zero_grad()

            features = model(batch_g, batch_x)
            outputs = classifier_model_1(features).squeeze()

            loss_cls = CRITERION_CLS(outputs, batch_label) * LAMBDA
            loss_mse = CRITERION_MSE(features.squeeze(), batch_z) 
            loss = loss_cls + loss_mse
            loss.backward()  
            OPTIMIZER.step()

            train_loss_mse += loss_mse.item()
            train_loss_ce += loss_cls.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_label.size(0)
            correct_train += (predicted == batch_label).sum().item()
            progress_bar.set_postfix_str(
                f'train_loss_mse={train_loss_mse / (i + 1):.4f}, train_loss_ce={train_loss_ce / (i + 1):.4f} \
, train_acc={100 * correct_train / total_train:.4f}')


        train_losses_mse.append(train_loss_mse/len(TRAIN_DATALOADER))
        train_losses_ce.append(train_loss_ce/len(TRAIN_DATALOADER))
        train_accs.append(100 * correct_train / total_train)

        model.eval()
        test_loss_mse = 0.0
        test_loss_ce = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            progress_bar_test = tqdm.tqdm(enumerate(TEST_DATALOADER), total=len(TEST_DATALOADER), desc=f'test set')

            for i,(batch_x, batch_z, batch_g, batch_y) in progress_bar_test:
                
                batch_z = batch_z.to(device)
                batch_g = batch_g.to(device)
                batch_x = batch_x.to(device).permute(0,2,1)
                batch_label = batch_y.to(device)

                features = model(batch_g, batch_x)
                outputs = classifier_model_1(features).squeeze()

                loss_cls = CRITERION_CLS(outputs, batch_label) * LAMBDA
                loss_mse = CRITERION_MSE(features.squeeze(), batch_z)

                test_loss_mse += loss_mse.item()
                test_loss_ce += loss_cls.item()
                _, predicted = torch.max(outputs, 1)
                total_test += batch_label.size(0)
                correct_test += (predicted == batch_label).sum().item()

                progress_bar_test.set_postfix_str(
                    f'test_loss_mse={test_loss_mse / (i + 1):.4f}, test_loss_ce={test_loss_ce / (i + 1):.4f} \
, test_acc={100 * correct_test / total_test:.4f}')
            
            test_losses_mse.append(test_loss_mse/len(TEST_DATALOADER))
            test_losses_ce.append(test_loss_ce/len(TEST_DATALOADER))
            test_accs.append(100 * correct_test / total_test)


        model.metrics_now = {
                    'train_loss_mse': -train_losses_mse[-1],
                    'train_loss_ce': -train_losses_ce[-1],
                    'train_acc': train_accs[-1],
                    'test_acc': test_accs[-1],
                    'test_loss_mse': -test_losses_mse[-1],
                    'test_loss_ce': -test_losses_ce[-1],
        } 

        mlflow.log_metric('train_loss_mse', train_losses_mse[-1], step=epoch)
        mlflow.log_metric('test_loss_mse', test_losses_mse[-1], step=epoch) 
        mlflow.log_metric('train_loss_ce', train_losses_ce[-1], step=epoch)
        mlflow.log_metric('test_loss_ce', test_losses_ce[-1], step=epoch) 
        mlflow.log_metric('train_acc', train_accs[-1], step=epoch)
        mlflow.log_metric('test_acc', test_accs[-1], step=epoch) 

        if EARLY_STOPPING2 == 'test_acc':
            do_break = model.early_stopping(test_accs[-1],epoch)
        elif EARLY_STOPPING2 == 'test_loss_mse':
            do_break = model.early_stopping(-test_losses_mse[-1],epoch)
        elif EARLY_STOPPING2 == 'test_loss_ce':
            do_break = model.early_stopping(-test_losses_ce[-1],epoch)
        elif EARLY_STOPPING2 == 'train_acc':
            do_break = model.early_stopping(train_accs[-1],epoch)
        elif EARLY_STOPPING2 == 'train_loss_mse':
            do_break = model.early_stopping(-train_losses_mse[-1],epoch)
        elif EARLY_STOPPING2 == 'train_loss_ce':
            do_break = model.early_stopping(-train_losses_ce[-1],epoch)

        if do_break:
            break
        mlflow.log_artifacts('temp', 'artifacts')

    # Save confusion matrix
    model = torch.load(model.save_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        train_list = []
        test_list = []
        train_list_y = []
        test_list_y = []
        for batch_x, _, batch_g, batch_y in train_loader:
            label = batch_y.to(device)
            batch_x = batch_x.to(device).permute(0,2,1)
            batch_g = batch_g.to(device)

            features = model(batch_g, batch_x)
            outputs = classifier_model_1(features) 

            _, outputs_cls = torch.max(outputs.squeeze(1),1) 
            train_list_y.append(label)
            train_list.append(outputs_cls)

        for batch_x, _, batch_g, batch_y  in test_loader:
            label = batch_y.to(device)
            batch_x = batch_x.to(device).permute(0,2,1)
            batch_g = batch_g.to(device)

            features = model(batch_g, batch_x)
            outputs = classifier_model_1(features) 

            _, outputs_cls = torch.max(outputs.squeeze(1),1) 
            test_list_y.append(label)
            test_list.append(outputs_cls)

    train_cls = torch.concat(train_list, dim=0)
    test_cls = torch.concat(test_list, dim=0)
    train_cls_y = torch.concat(train_list_y, dim=0)
    test_cls_y = torch.concat(test_list_y, dim=0)

    cm_train = confusion_matrix(train_cls_y.cpu(), train_cls.cpu())
    cm_test = confusion_matrix(test_cls_y.cpu(), test_cls.cpu())
    report_train = classification_report(train_cls_y.cpu(), train_cls.cpu())
    report_test = classification_report(test_cls_y.cpu(), test_cls.cpu())

    with open('temp/cm_train_R.pkl', 'wb') as file:
        pickle.dump(cm_train, file)
    with open('temp/cm_test_R.pkl', 'wb') as file:
        pickle.dump(cm_test, file)
    
    with open('temp/report_train_R.txt', 'w') as file:
        file.write(report_train)
    with open('temp/report_test_R.txt', 'w') as file:
        file.write(report_test)
    mlflow.log_artifacts('temp', 'artifacts')





