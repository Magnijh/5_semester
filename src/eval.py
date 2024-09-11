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

# Change the parameters in the read_csv() function to match your csv-file, this is the data the model will be evaluated on.
# If the dataset is too big it is advised that you skip a couple of rows, since memory might become an issue. Also if done on a CPU this process can take a while.
df = pd.read_csv("CSV_Combiner\mergedData24h.csv", header=0, names=["date", "Crude Oil", "Dollar"], skiprows=round(11062*0.8))
# df = pd.read_csv("CSV_Combiner\mergedData30m.csv", header=0, names=["date", "Crude Oil", "Dollar"], skiprows=round(16746*0.8))
df.fillna(method='ffill', inplace=True)

# Change the first parameter in _loadModel() to name of the model you want to evaluate 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, scaler, in_seq_length, seq_length = pre._loadModel("24h-In3-Out1_E75_loss_0.02732", device)
scaler = util.StandardScaler(
    mean=torch.tensor(df.iloc[:, 1:].mean(0), device=device, dtype=torch.float).reshape(1, 1, -1, 1),  #
    std=torch.tensor(df.iloc[:, 1:].std(0), device=device, dtype=torch.float).reshape(1, 1, -1, 1))  #

# The 1 in df.iloc[] skips the first column in the dataset (in our case the first column contains the date)
X, Y = generate_graph_seq2seq_io_data( df.iloc[:, 1:], in_seq_length, seq_length)

outputs_x = []
outputs_y = []

testx = torch.Tensor(X).to(device)
testx = testx.transpose(1, 3)
with torch.no_grad():
    preds = model(scaler.transform(testx)).transpose(1, 3)
outputs_x.append(preds)
Y = torch.tensor(Y, device=device).transpose(1, 3)
outputs_y.append(Y)  #

yhat = torch.cat(outputs_x, dim=0)
realy = torch.cat(outputs_y, dim=0)
#realy = scaler.transform(realy)
yhat = scaler.inverse_transform(yhat)
yhat = yhat[:realy.size(0)]

Y_real = realy[:, 0, :, -1]
Y_pred = yhat[:, 0, :, -1]
# The second parameter in plt.subplots() has to equal to the number of features
fig, axs = plt.subplots(1, 2, dpi=180, figsize=(8, 4))
for i, ax in enumerate(axs):
    ax.plot(Y_real[:, i].detach().cpu().numpy(), label="real")
    ax.plot(Y_pred[:, i].detach().cpu().numpy(), label="pred")
    ax.set_title(df.columns[i + 1])
plt.legend()

pred = yhat
real = realy
metrics = util.metric(pred, real)
log = 'MAE: {:.8f}, MAPE: {:.8f}, RMSE: {:.8f}'
print(log.format(metrics[0], metrics[1], metrics[2]))