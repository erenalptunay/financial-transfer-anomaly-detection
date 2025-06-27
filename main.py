import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y_true = df['Class']

