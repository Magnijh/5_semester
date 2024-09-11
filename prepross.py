#%%
import pandas as pd
from util import StandardScaler, DataLoader
import torch

# %%
df = pd.read_csv("CSV_Combiner\mergedData30m.csv", header=0, names=["date", "Oil", "Dollar"])
df.fillna(method='ffill', inplace=True)

#%%

import torch
import numpy as np
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

from generate_training_data import generate_graph_seq2seq_io_data


INPUT_LENGTH = 3
SEQ_LENGTH = 36  #H - History Length (out dim/length)
BATCH_SIZE = 64
device = torch.device('cuda')
x_offsets = np.sort(np.concatenate((np.arange(-(INPUT_LENGTH - 1), 1, 1),)))
y_offsets = np.sort(np.arange(1, (SEQ_LENGTH + 1), 1))
# x: (num_samples, input_length, num_nodes, input_dim)
# y: (num_samples, output_length, num_nodes, output_dim)
X, Y = generate_graph_seq2seq_io_data(
    df.iloc[:,1:],
    x_offsets=x_offsets,
    y_offsets=y_offsets,
    add_time_in_day=False,
    add_day_in_week=False,
)

dataloader = {}
dataloader['train_loader'] = DataLoader(X[0:int(X.shape[0] * 0.8)], Y[0:int(Y.shape[0] * 0.8)], BATCH_SIZE)
dataloader['test_loader'] = DataLoader(X[int(X.shape[0] * 0.8):int(X.shape[0] * 0.9)],
                                       Y[int(Y.shape[0] * 0.8):int(Y.shape[0] * 0.9)],
                                       BATCH_SIZE)
dataloader['val_loader'] = DataLoader(X[int(X.shape[0] * 0.9):X.shape[0]], Y[int(Y.shape[0] * 0.9):Y.shape[0]], BATCH_SIZE)
scaler = StandardScaler(
    mean=torch.tensor(df.iloc[:, 1:].mean(0), device=device, dtype=torch.float).reshape(1, 1, -1, 1),  #
    std=torch.tensor(df.iloc[:, 1:].std(0), device=device, dtype=torch.float).reshape(1, 1, -1, 1))  #
dataloader['scaler'] = scaler
#%%
IN_DIM = 1  #NUMBER OF TIMESERIES
NUM_NODES = 2  #NUMBER OF FEATURES
NHID = 256
DROPOUT = 0.3
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.001

engine = trainer(scaler,
                 IN_DIM,
                 SEQ_LENGTH,
                 NUM_NODES,
                 NHID,
                 DROPOUT,
                 LEARNING_RATE,
                 WEIGHT_DECAY,
                 device,
                 None,
                 True,
                 True,
                 None)

print("start training...", flush=True)
his_loss = []
val_time = []
train_time = []
EPOCHS = 10
SAVE = './garage/metr'
EXPID = 0
PRINT_EVERY = 10
for i in range(1, EPOCHS + 1):
    train_loss = []
    train_mape = []
    train_rmse = []
    t1 = time.time()
    dataloader['train_loader'].shuffle()
    engine.model.train()
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainx = trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        metrics = engine.train(trainx, trainy[:, 0, :, :])
        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
        if iter % PRINT_EVERY == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
    t2 = time.time()
    train_time.append(t2 - t1)

    #validation
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    
    s1 = time.time()
    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        metrics = engine.eval(testx, testy[:, 0, :, :])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    s2 = time.time()
    log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
    print(log.format(i, (s2 - s1)))
    val_time.append(s2 - s1)
    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)
    
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
    
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
    torch.save(engine.model.state_dict(), SAVE + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

#testing
bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load(SAVE + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

engine.model.eval()
outputs_x = []
outputs_y = []
for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    with torch.no_grad():
        preds = engine.model(scaler.transform(testx)).transpose(1, 3)
    outputs_x.append(preds)
    y = torch.tensor(y, device=device).transpose(1, 3)
    outputs_y.append(y)  #

yhat = torch.cat(outputs_x, dim=0)
realy = torch.cat(outputs_y, dim=0)
yhat = scaler.inverse_transform(yhat)
yhat = yhat[:realy.size(0)]
print("Training finished")
print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

amae = []
amape = []
armse = []
for i in range(SEQ_LENGTH):
    pred = yhat[..., i]
    real = realy[..., i]
    metrics = util.metric(pred, real)
    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
    amae.append(metrics[0])
    amape.append(metrics[1])
    armse.append(metrics[2])

log = 'On average over the horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
torch.save(engine.model.state_dict(), SAVE + "_exp" + str(EXPID) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
torch.save({"mean": dataloader['scaler'].mean, "std": dataloader['scaler'].std},
            SAVE + "_exp" + str(EXPID) + "_best_" + str(round(his_loss[bestid], 2)) + "_SCALER.pth")
#%%
Y_real = realy[:, 0, :, -1]
Y_pred = yhat[:, 0, :, -1]
#%%
fig, axs = plt.subplots(1, 2, dpi=180, figsize=(8, 4))
for i, ax in enumerate(axs):
    ax.plot(Y_real[:, i].detach().cpu().numpy(), label="real")
    ax.plot(Y_pred[:, i].detach().cpu().numpy(), label="pred")
    ax.set_title(df.columns[i + 1])
plt.legend()


#%%
def predict(x, NN=NUM_NODES, SL=SEQ_LENGTH, IDM=IN_DIM):
    with torch.no_grad():
        X_preds = engine.model(scaler.transform(x.unfold(0, SEQ_LENGTH, 1).unsqueeze(1))).transpose(1, 3)
    return scaler.inverse_transform(X_preds)[:, 0, :, -1]


# %%
X_test = realy[:, 0, :, -1].float()
Y_test = predict(X_test)
XY_diff = X_test.shape[0]-Y_test.shape[0]
# %%
fig, axs = plt.subplots(1, 2, dpi=180, figsize=(8, 4))
for i, ax in enumerate(axs):
    ax.plot(X_test[XY_diff:, i].detach().cpu().numpy(), label="real")
    ax.plot(Y_test[:X_test.size(0), i].detach().cpu().numpy(), label="pred")
    ax.set_title(df.columns[i + 1])
plt.legend()
# %%
fig, axs = plt.subplots(1, 2, dpi=180, figsize=(8, 4))
for i, ax in enumerate(axs):
    ax.plot(((X_test[XY_diff:, i])-(Y_test[:, i])).detach().cpu().numpy())
    ax.set_title(df.columns[i + 1])
plt.legend()
# %%
