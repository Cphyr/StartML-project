# load patient_survival.csv
import pandas as pd
import numpy as np

df = pd.read_csv('data/patient_survival.csv')

print(df.head())
print(df.columns)