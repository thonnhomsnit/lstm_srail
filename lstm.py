import csv
import numpy as np
import pandas as pd

x = pd.read_csv(r'C:/Users/Personal/Documents/GitHub/lstm_srail/matlabinput.csv')
y=2*x.values

with open('pythonoutput.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(y)

