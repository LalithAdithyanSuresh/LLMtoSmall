import model_trainer
import getGeminiResponse as gemini
import prompts
import os
import json
import getData

print(gemini.getResponse("Give me only a JSON just like this {'message':'hi'}")['message'])


def doTask(task,taskFile):
    taskManaged = os.listdir("tasks")
    dataSetInfo = getData.onlyHeaders(taskFile)
    taskDetails = gemini.getResponse(prompts.TASK_CONFIRMATION.format(task,str(taskManaged),str(dataSetInfo)))
    if taskDetails["same"] == 0:
        fileName = str(taskDetails["taskId"]) + "_"+ taskDetails["taskName"].replace(" ","-") +".json"
        taskFile = getData.cleanDataFrame(taskFile,taskDetails['datasetClean'],fileName)
        taskPath = "tasks/"+fileName
        newFile = open(taskPath,"w")
        with newFile:
            content = {
                "taskId": taskDetails["taskId"],
                "taskName": taskDetails["taskName"],
                "fileName": taskFile,
                "rows_predicted" : 0,
                "best_model_metricts" : None,
                "rows_used" : []
            }
            json.dump(content, newFile,indent=4)
        newFile.close()

        rowsToTrain = getData.fromFile(taskFile)
        print(model_trainer.train_model(rowsToTrain))
        






doTask("Get all spam emails","../data/emails_shuffled.csv")