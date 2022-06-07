import pandas as pd
import numpy as np

pd = df = pd.read_csv("arg_busan.csv")
inputColumn = ["tide_cm", "wat_temp_c"]
print(df[inputColumn])
inputFeatures = np.array(df[inputColumn])
print(inputFeatures)
