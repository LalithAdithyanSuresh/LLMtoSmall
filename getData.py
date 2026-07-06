from pandas.io import json
import pandas as pd
import random
data_base ="dataPool/"

def onlyHeaders(file_path):
    df = pd.read_csv(file_path, nrows=5)
    return str({col: str(dtype) for col, dtype in df.dtypes.to_dict().items()})

def cleanDataFrame(file_path, command,newName):
    df = pd.read_csv(file_path)
    
    local_vars = {'df': df}
    exec(command, globals(), local_vars)
    
    updated_df = local_vars['df']
    
    updated_df.to_csv(data_base + newName.replace(".json",".csv"), index=False)
    return newName.replace(".json",".csv")

def fromFile(fileName,startRow = 0,offset = 200):
    df = pd.read_csv(data_base + fileName, skiprows=range(1, startRow), nrows=offset)
    return df
    

def fromTask(filePath):
    with open(filePath, 'r') as file:
        return json.load(file)