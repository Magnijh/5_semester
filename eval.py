import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import prediction_backend as pre
import util

def generate_graph_seq2seq_io_data(df, in_length, out_length):
    x_offsets = np.sort(np.concatenate((np.arange(-(in_length - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (out_length + 1), 1))
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

#df = pd.read_csv("OK_EU.csv", header=0, names=["date", "OK", "EU"], skiprows=400)
#df = pd.read_csv("CSV_Combiner\mergedData30m.csv", header=0, names=["date", "Crude Oil", "Dollar"], skiprows=round(16746*0.8))
df = pd.read_csv("CSV_Combiner\mergedData24h.csv", header=0, names=["date", "Crude Oil", "Dollar"], skiprows=7753)

df.fillna(method='ffill', inplace=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model, scaler, seq_length = pre._loadModel("Oil_dolar_in3_out9_E10_addaptadj-True", device)
# model, scaler, seq_length = pre._loadModel("Oil_dolar_in3_out18_E10_addaptadj-True", device)
model, scaler, seq_length = pre._loadModel("Oil_dolar_in3_out36_E10_addaptadj-True", device)
scaler = util.StandardScaler(
    mean=torch.tensor(df.iloc[:, 1:].mean(0), device=device, dtype=torch.float).reshape(1, 1, -1, 1),  #
    std=torch.tensor(df.iloc[:, 1:].std(0), device=device, dtype=torch.float).reshape(1, 1, -1, 1))  #

#TODO add input length to function
X, Y = generate_graph_seq2seq_io_data( df.iloc[:, 1:], 10, seq_length)

outputs_x = []
outputs_y = []

testx = torch.Tensor(X).to(device)
testx = testx.transpose(1, 3)
with torch.no_grad():
    preds = model(scaler.transform(testx)).transpose(1, 3)
outputs_x.append(preds)
y = torch.tensor(Y, device=device).transpose(1, 3)
outputs_y.append(y)  #

yhat = torch.cat(outputs_x, dim=0)
realy = torch.cat(outputs_y, dim=0)
yhat = scaler.inverse_transform(yhat)
yhat = yhat[:realy.size(0)]

Y_real = realy[:, 0, :, -1]
Y_pred = yhat[:, 0, :, -1]
fig, axs = plt.subplots(1, 2, dpi=180, figsize=(8, 4))
for i, ax in enumerate(axs):
    ax.plot(Y_real[:, i].detach().cpu().numpy(), label="real")
    ax.plot(Y_pred[:, i].detach().cpu().numpy(), label="pred")
    ax.set_title(df.columns[i + 1])
plt.legend()

pred = yhat
real = realy
metrics = util.metric(pred, real)
log = 'MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
print(log.format(metrics[0], metrics[1], metrics[2]))