# Crude Oil Forecasting

### Step by step guide
* Select the input format and provide the input that is needed
* Select a model (beneficial to select one trained on the same time interval as the provided data)
* Press visualite results
* Optional: specify the predictions length

### The Microservice
This microservice aims to predict the price of crude oil in U.S. dollars based on previous data. The model is made using PyTorch utilizing the Graph WaveNet algorithm. the microservice contain multiple models trained on different datasets where the difference between the data is the time interval between each data entry. The user selects what model they want to use by selecting it in a dropdown menu. The microservice will output a graph that includes the data points and the newly predicted price, as well as a table with the same information. 

### User Interactions
The users of the microservice have three different ways of using the service. If you select text you can provide the data in text seperated by commas. If you select file you can upload a csv (comma-seperated) where the first column contains the oil price and the second column contains the dollar index. Finally the example data option allows the user to test out the service, without having to provide any data.

Below the input fields, there is a dropdown menu where the user can choose which models they want to use their data on. The user should aim to use the model closest to their data which means that if the time between each data entry is thirty minutes the user should use the model with thirty minutes between each data entry. 

The naming of the models can be split into three components, the time interval, the input length and the prediction length. They are formatted in this manner: timeInterval-inputLength-predictionLength.

Lastly, you can specify the amount of predictions outputted by the service. Has to be an integer and within the range 1-256.

### The Data
The microservice takes as input a list that needs to have a certain data structure. There is a list that contains the oil prices in U.S. Dollar and a list which contains the dollar indexes. These lists are combined so that: data = [oilPricesList, dollarIndexesList]. The list is then made into a tensor and used on the model to predict. 

### Data service route
The data service route allows calls for the data that the models are trained on. 
Below is the different ways to call for data service route.
* /data?src/filename.csv?Sstartdate?Eenddate
* /data?src/filename.csv?Sstartdate
* /data?src/filename.csv?Eenddate
* filename: mergedData24h or mergedData30min
* timeformat: year/month/day-hour:min

Example request for data:
/data?src/mergedData24h.csv?S1896/01/02-00:00?E1986/01/10-00:00 

### Making a new model
To those interested in making a new model we have provided comments in the following py-files, that explain what you have to do. The file responsible for making new models is prepross.py, eval.py can then be utilized to evalute the new model. To test predictions you can use predTest.py and finally, if you want to add a new model to our service visit prediction_backend.py. 

Also, if you have multiple csv files you can use the CSV combiner to merge csv-files on the date. Note that the csv combiner might dump some data to the bottom of the csv, which should be manually deleted.
<a href="https://wiki.astep-dev.cs.aau.dk/services/CrudeOilPrediction" target="_blank"> The aSTEP wiki page. </a>
