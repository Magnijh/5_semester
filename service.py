# Import of libraries needed for the service to run on the aSTEP website
from flask import Flask, jsonify, request, Response, json
# Cors is used so the docker containers can to talk together
from flask_cors import CORS
import pandas as pd
import src.prediction_backend as pb


# Basic configuration
# Service is the name we use to access the application as a whole.
service = Flask(__name__)
cors = CORS(service)
#service.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
service.config['CORS_HEADERS'] = 'Content-Type'

# The '/' is the index of the page.
def printText(content):
    return jsonify({
        'chart_type': 'markdown',
        'content': content
    })


@service.route('/')
def hello():
    return "try /info"

# The info tells where to place the service in the outer menu (category element) and what to name it.


@service.route('/info')
def info():
    return jsonify({
        'id': 'Crude_Oil_Price',
        'name': 'Crude oil prediction',
        'version': '2019',
        'category': 2,
    })

# fields represents the input of the service. This is where input "parameters" are specified and named.


@service.route('/fields')
def fields():

    models = [{'name': x.modelName, 'value': x.modelName} for x in pb.Models]  
    timeIntervals = [{'name': '30 minute interval', 'value': 'thirtyMinInterval'}, {'name': '24 hour interval', 'value': 'oneDayInterval'}]

    return jsonify({
        'user_fields': [
            {
                "type": "formset-select",
                "name": "inputOption",
                "label": "Select input format:",
                "options": [
                    {
                        "name": "Example data",
                        "value": "example",
                        "fields": [
                            {
                                "type": "select",
                                "name": "timeInterval",
                                "label": "Select time interval:",
                                "options": timeIntervals
                            }
                        ]
                    },
                    {
                        "name": "Text",
                        "value": "text",
                        "fields": [
                            {
                                'type': 'input',
                                'name': 'oilPrices',
                                'label': 'Write the most recent oil prices:',
                                'placeholder': '75.44, 75.06, 75.22, 76.01, 76.01'
                            },
                            {
                                'type': 'input',
                                'name': 'dollarIndexes',
                                'label': 'Write the most recent dollar indexes:',
                                'placeholder': '93.41, 93.72, 94.34, 94.30, 94.30'
                            }
                        ]
                    },
                    {
                        "name": "File",
                        "value": "csv",
                        "fields": [
                            {
                                "type": "file",
                                "name": "csvFile",
                                "label": "Upload a csv-file:",
                                "placeholder": "..."
                            }
                        ]
                    }
                ]
            },
            {
                "type": "select",
                "name": "model",
                "label": "Select model:",
                "options": models
            },
            {
                    "type": "input-number",
                    "name": "predictionLength",
                    "label": "Prediction length:",
                    "placeholder": "10",
                    "help_text": "Not necessary to specify"
            }  
        ],
        'developer_fields': [
        ]
    })

# The readme is loaded when the documentation is loaded.


@service.route('/readme')
def readme():
    with open('README.md', 'r') as readMeFile:
        readMeContent = readMeFile.read()
    return jsonify({
        'chart_type': 'markdown',
        'content': readMeContent
    })


# Combined endpoint, neeeded for notebook mode. Read the wiki for more information
@service.route('/combined', methods=['POST'])
def combine():
    return jsonify({
        'render': render(),
        'data': data()
    })

# This function is called when the 'visualize results' is pressed.


@service.route('/render', methods=['GET', 'POST'])
def render():
    inputOption = request.form.get('inputOption')
    selectedModel = request.form.get('model')
    selectedInterval = request.form.get('timeInterval')
    predictionLength = request.form.get('predictionLength')
    oilPricesList = []
    dollarIndexesList = []

    if inputOption == 'example':
        if selectedInterval == '':
            return printText('Select a time interval.')
    elif inputOption == 'text' and request.form.get('oilPrices') != '' and request.form.get('dollarIndexes') != '':
        oilPrices = request.form.get('oilPrices')
        dollarIndexes = request.form.get('dollarIndexes')
        try:
            oilPricesList = [float(x) for x in oilPrices.split(',')]
            dollarIndexesList = [float(x) for x in dollarIndexes.split(',')]
        except ValueError:
            return printText('The type of the given input is incorrect.')
    elif inputOption == 'csv' and 'csvFile' in request.files:
        file = request.files['csvFile']
        fileType = file.filename.split('.')[1].lower()
        if (fileType != 'csv'): 
            return printText('The file given is not a csv-file.')
        df = pd.read_csv(file, header = None)
        try:
            oilPricesList = list(df.iloc[:,0].astype(float))
            dollarIndexesList = list(df.iloc[:,1].astype(float))
        except ValueError:
            return printText('The type of the data in the csv-file is incorrect.')
    elif inputOption == '':
        return printText('Select an input format.')

    modelNames = [x.modelName for x in pb.Models]

    if selectedModel in modelNames:
        for x in pb.Models:
            if x.modelName == selectedModel:
                model = x
                break

        if inputOption == 'example' and selectedInterval == 'oneDayInterval':
            df = pd.read_csv('src/mergedData24h.csv', skiprows = 11062 - model.in_seq_length - 1)
            oilPricesList = list(df.iloc[:,1])
            dollarIndexesList = list(df.iloc[:,2])
        elif inputOption == 'example' and selectedInterval == 'thirtyMinInterval':
            df = pd.read_csv('src/mergedData30min.csv', skiprows = 16746 - model.in_seq_length - 1)
            oilPricesList = list(df.iloc[:,1])
            dollarIndexesList = list(df.iloc[:,2])

    
        if (len(oilPricesList) != model.in_seq_length or len(dollarIndexesList) != model.in_seq_length):
            return printText(f'The list of inputs is not exactly {model.in_seq_length} long in both instances.')
        
        inputData = [oilPricesList, dollarIndexesList]

        if predictionLength == '':
            predictionLength = model.seq_length
        else:
            try:
                predictionLength = int(predictionLength) 
                if not (predictionLength >= 1 and predictionLength <= 256):
                    return printText('Prediction length has to be in the range 1-256.')
            except: 
                return printText('Prediction length is not an integer.')

        combinedData = pb.predict(model.modelName, inputData, predictionLength)
        combinedOilPrices = combinedData[0]
        combinedDollarIndexes = combinedData[1]
        timeSteps = []

        for i in range(model.in_seq_length + predictionLength):
            timeSteps.append(f'{i - model.in_seq_length + 1}')
        
        oilPricesRowList = [combinedData[0], timeSteps] 
        dollarIndexesRowList = [combinedData[1], timeSteps] 

        return jsonify({
            'chart_type': 'composite',
            'content': [
                {
                    'name': 'Oil prices',
                    'chart_type': 'composite-scroll',
                    'content': [
                        {
                            'chart_type': 'markdown',
                            'content': 'The graph shows the oil price as a function over time steps and the table below contains the data points from the graph. The data points at the time steps up until 0 are the input oil prices, the ones that come afterwards are the predicted oil prices.'
                        },
                        {
                            'chart_type':'chart-js',
                            'content': {
                                'data': [
                                    {
                                        'data': combinedOilPrices,
                                        'label': 'Oil prices'
                                    }
                                ],
                                'labels': timeSteps,          
                                'colors': [
                                    {
                                        'backgroundColor': 'rgba(252, 221, 20, 0.1)',
                                        'borderColor': 'rgb(252, 221, 20)',
                                        'pointBackgroundColor': 'rgb(252, 126, 0)',
                                        'pointBorderColor': '#fff',
                                        'pointHoverBackgroundColor': '#fff',
                                        'pointHoverBorderColor': 'rgb(252, 126, 0)'
                                    }
                                ],
                                'type': 'line'
                            }
                        },
                        {
                            'chart_type':'simple-table',
                            'content': {
                                'settings': [
                                    {
                                        'objectKey': 'time',
                                        'columnOrder': 0
                                    },
                                    {
                                        'objectKey': 'oilPrice',
                                        'columnOrder': 1
                                    }
                                ],
                                'fields': [
                                    {
                                        'name': 'Time steps',
                                        'objectKey': 'time'
                                    },
                                    {
                                        'name': 'Oil price',
                                        'objectKey': 'oilPrice'
                                    }
                                ],
                                'data': [{'time': oilPricesRowList[1][x], 'oilPrice': str("{:.8f}".format(oilPricesRowList[0][x])) + ' $'} for x in range(len(oilPricesRowList[0]))]
                            }
                        }
                    ]
                },
                {
                    'name':'Dollar indexes',
                    'chart_type': 'composite-scroll',
                    'content': [
                        {
                            'chart_type': 'markdown',
                            'content': 'The graph shows the dollar index as a function over time steps and the table below contains the data points from the graph. The data points at the time steps up until 0 are the input dollar indexes, the ones that come afterwards are the predicted dollar indexes.'
                        },
                        {
                            'chart_type':'chart-js',
                            'content': {
                                'data': [
                                    {
                                        'data': combinedDollarIndexes,          
                                        'label': 'Dollar indexes'            
                                    }
                                ],
                                'labels': timeSteps,          
                                'colors': [
                                    {
                                        'backgroundColor': 'rgba(252, 221, 20, 0.1)',
                                        'borderColor': 'rgb(252, 221, 20)',
                                        'pointBackgroundColor': 'rgb(252, 126, 0)',
                                        'pointBorderColor': '#fff',
                                        'pointHoverBackgroundColor': '#fff',
                                        'pointHoverBorderColor': 'rgb(252, 126, 0)'
                                    }
                                ],
                                'type': 'line'
                            }
                        },
                        {
                            'chart_type':'simple-table',
                            'content': {
                                'settings': [
                                    {
                                        'objectKey': 'time',
                                        'columnOrder': 0
                                    },
                                    {
                                        'objectKey': 'dollarIndex',
                                        'columnOrder': 1
                                    }
                                ],
                                'fields': [
                                    {
                                        'name': 'Time steps',
                                        'objectKey': 'time'
                                    },
                                    {
                                        'name': 'Dollar index',
                                        'objectKey': 'dollarIndex'
                                    }
                                ],
                                'data': [{'time': dollarIndexesRowList[1][x], 'dollarIndex': "{:.8f}".format(dollarIndexesRowList[0][x])} for x in range(len(dollarIndexesRowList[0]))]
                            }
                        }
                    ]
                }
            ]
        })
    elif selectedModel == '':
        return printText('Select a model.')

    return readme()


# This shows the raw data of the output in raw JSON format.
@service.route("/data",methods=['GET','POST'])
def data():
    pathFromUrl = request.full_path
    csvFiles = ['src/mergedData24h.csv', 'src/mergedData30min.csv']
    splittingAtQuestionmark = pathFromUrl.split("?",3)
    lengthOfSplitAtQuestiomark = len(splittingAtQuestionmark)

    if splittingAtQuestionmark[1] in csvFiles:
        print(splittingAtQuestionmark[1][0])
        fileName = splittingAtQuestionmark[1]
    else:
        return "Incorrect parameters."

    if lengthOfSplitAtQuestiomark > 2 and splittingAtQuestionmark[2][0] == "S":
        startTime = splittingAtQuestionmark[2][1:]
        startTime = startTime.replace("-", " ")
        startTime = pd.to_datetime(startTime)
    else:
        startTime = None

    if (lengthOfSplitAtQuestiomark > 3 and startTime is not None and splittingAtQuestionmark[3][0] == "E") or (lengthOfSplitAtQuestiomark > 2 and startTime is not None and splittingAtQuestionmark[2][0] == 'E'):
        endTime = splittingAtQuestionmark[3][1:]
        endTime = endTime.replace("-", " ")
        endTime = pd.to_datetime(endTime)
    else:
        endTime = None

    csvFileRead = pd.read_csv(fileName)
    csvFileRead['Date'] = pd.to_datetime(csvFileRead['Date'])

    if startTime is None and endTime is None:
        return csvFileRead.to_json()
    elif startTime is not None and endTime is None:
        lastRow = csvFileRead.iloc[-1]
        endTime = lastRow[0]
        mask = (csvFileRead['Date'] > startTime) & (csvFileRead['Date'] <= endTime)
        csvFileRead = csvFileRead.loc[mask]
        return csvFileRead.to_json()
    elif startTime is None and endTime is not None:
        firstrow = csvFileRead.iloc[0]
        startTime = firstrow[0]
        mask = (csvFileRead['Date'] > startTime) & (csvFileRead['Date'] <= endTime)
        csvFileRead = csvFileRead.loc[mask]
        return csvFileRead.to_json()
    elif startTime is not None and endTime is not None:
        mask = (csvFileRead['Date'] > startTime) & (csvFileRead['Date'] <= endTime)
        csvFileRead = csvFileRead.loc[mask]
        return csvFileRead.to_json()
    else:
        return 'Data was not retrieved.'
        


# Likewise for further information of the above used functions.
if __name__ == '__main__':
    service.run(host='0.0.0.0', port=5000)
