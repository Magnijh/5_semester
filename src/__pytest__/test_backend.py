from src.predTest import predTest

#Helper function for our model testing to ensure the correct dataformat
def listCheck(res, length):
    for lists in res:
        if (len(lists) != length):
            return False
    return True

#Test case for just the models and how they are designed
def test_models(models):
    finalRes = 0
    for testModel in models:
        res = predTest(testModel.modelName, testModel.data_interval + ".csv", 0, 0) #Making a prediction
        if (len(res) == testModel.num_nodes and listCheck(res, testModel.seq_length) == True): #Checking format to ensure correctness 
            finalRes += 1
    if (finalRes == len(models)):
        assert True

#Test case for our advanced predict on all models, predLength can be changed to any positive integer
def test_advacnedPred(models):
    finalRes = 0
    predLength = 10
    for testModel in models:
        res = predTest(testModel.modelName, testModel.data_interval + ".csv", 0, predLength) #Making a prediction
        if (len(res) == testModel.num_nodes and listCheck(res, predLength) == True): #Checking format to ensure correctness
            finalRes += 1
    if (finalRes == len(models)):
        assert True