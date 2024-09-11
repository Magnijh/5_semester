import torch
import math

from src.util import StandardScaler
from src.engine import trainer

class Model:
    def __init__(self, modelName, in_seq_length, seq_length, in_dim, num_nodes, nhid, dropout, lerning_rate, weight_decay, addaptadj, data_interval):
        self.modelName = modelName
        self.in_seq_length = in_seq_length
        self.seq_length = seq_length
        self.addaptadj = addaptadj
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.nhid = nhid
        self.dropout = dropout
        self.lerning_rate = lerning_rate
        self.weight_decay = weight_decay
        self.data_interval = data_interval

# New models have to be added to this list of models, insert the value of the variables from prepross when the model was trained
# In order to get the model to load it has to be added to the models folder in src along with its scalar
Models=[Model("24h-In3-Out1", 3, 1, 1, 2, 256, 0.3, 0.001, 0.0001, True, "24h"), # 32 min training
        Model("24h-In5-Out7", 5, 7, 1, 2, 256, 0.3, 0.001, 0.0001, True, "24h"), # 35 min training
        Model("30min-In3-Out4", 3, 4, 1, 2, 256, 0.3, 0.001, 0.0001, True, "30min"), # 46.5 min training
        Model("30min-In5-Out48", 5, 48, 1, 2, 256, 0.3, 0.001, 0.0001, True, "30min")] # 44.5 min training

def predict(modelName, knownData, predictionLength=0, default_scaler=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, scaler, in_seq_length, seq_length = _loadModel(modelName, device)

    if not default_scaler: # calculate scaler based on input date and overwrite model scaler
        df = torch.tensor(knownData, device=device).reshape(1, 1, 2, -1)
        scaler = StandardScaler(
            mean=df.transpose(1,3).squeeze().mean(0).reshape(1, 1, -1, 1),  #
            std=df.transpose(1,3).squeeze().std(0).reshape(1, 1, -1, 1))  #

    if predictionLength == 0:
        return _simplePrediction(knownData, model, scaler, device)
    else:
        return _advancedPrediction(knownData, model, scaler, device, in_seq_length, seq_length, predictionLength)

def _loadModel(modelName, device):
    filePath, in_seq_length, seq_length, in_dim, num_nodes, nhid, dropout, lerning_rate, weight_decay, addaptadj, data_interval = _modelParameters(modelName)

    #load related scaler
    chkpt = torch.load(data_interval + "_SCALER.pth", map_location=device)
    mean = chkpt["mean"].to(device)
    std = chkpt["std"].to(device)
    scaler = StandardScaler(mean=mean, std=std)
    
    #load model
    model = trainer(scaler, in_dim, seq_length, num_nodes, nhid, dropout, lerning_rate, weight_decay, device, None, addaptadj, addaptadj, None).model
    model.load_state_dict(torch.load(filePath + ".pth", map_location=device))
    model.eval()

    return model, scaler, in_seq_length, seq_length

def _modelParameters(modelName):
    basePath = "src/models/"
    for x in Models:
        if x.modelName == modelName:
            return basePath + x.modelName, x.in_seq_length, x.seq_length, x.in_dim, x.num_nodes, x.nhid, x.dropout, x.lerning_rate, x.weight_decay, x.addaptadj, basePath + x.data_interval

    return basePath + x.modelName, x.in_seq_length, x.seq_length, x.in_dim, x.num_nodes, x.nhid, x.dropout, x.lerning_rate, x.weight_decay, x.addaptadj, basePath + x.data_interval

def _simplePrediction(x, model, scaler, device):
    df = torch.tensor(x, device=device).reshape(1, 1, 2, -1)
    
    with torch.no_grad():
        X_preds = model(scaler.transform(df)).transpose(1, 3)
    #scaler.mean[0, 0, 0, 0]-=5
    return _combineList(x, scaler.inverse_transform(X_preds)[0, 0, :, :].tolist())

def _advancedPrediction(x, model, scaler, device, in_seq_length, seq_length, predictionLength):
    df = scaler.transform(torch.tensor(x, device=device).reshape(1, 1, 2, -1))
    
    if predictionLength <= seq_length:
        repeat = 1
    else:
        repeat = math.ceil(predictionLength/seq_length)
    
    for i in range(repeat):
        with torch.no_grad():
            X_preds = model(df).transpose(1, 3)
        
        x = _combineList(x, scaler.inverse_transform(X_preds)[0, 0, :, :].tolist())
        df = X_preds[:, :, :, seq_length-in_seq_length:]

    return [sublist[:in_seq_length+predictionLength] for sublist in x]

def _combineList(L1, L2):
    for i in range(len(L1)):
        L1[i] += L2[i]
    return L1