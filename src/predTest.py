import src.prediction_backend as pb
import pandas as pd

def fetchData(datafile, amount, offset=0):
    df = pd.read_csv(datafile, header=0, names=["date", "Oil", "Dollar"])
    return [list(df.iloc[offset:offset+amount, 1]), list(df.iloc[offset:offset+amount, 2])]

def predTest(modelName, dataName, offset=0, predictionLength=0):
    basePath = "src/mergedData"
    filePath, in_seq_length, seq_length, in_dim, num_nodes, nhid, dropout, lerning_rate, weight_decay, addaptadj, data_interval = pb._modelParameters(modelName)
    data = fetchData(basePath + dataName, in_seq_length, offset)
    if predictionLength > 0:
        real = fetchData(basePath + dataName, in_seq_length+predictionLength, offset)
    else:
        real = fetchData(basePath + dataName, in_seq_length+seq_length, offset)

    result = pb.predict(modelName, data, predictionLength)

    #Uncomment if you want a plot of the data
    
    #import matplotlib.pyplot as plt
    #test = ["oil", "$"]
    #fig, axs = plt.subplots(1, 2, dpi=180, figsize=(8, 4))
    #for i, ax in enumerate(axs):
    #    ax.plot(result[i], label="pred")
    #    ax.plot(real[i], label="real")
    #    ax.set_title(test[i])
    #plt.legend()
    return result

# The first parameter is the name of the model you want to test
# The second paramater is the csv-file containing the data you want to predict from and compare to
# The third parameter is an optional offset when reading the csv-file
# The last parameter is an optional prediction Length specifier <--
# EXCLAIMER: The model has to be added to the list of models in prediction_backend.py and the models folder

#predTest("30m-In5-Out48_E75_loss_0.053125", "Data30m.csv", 600)