import pandas as pd
import numpy as np
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from treefarms.model.threshold_guess import compute_thresholds, cut
from treefarms import TREEFARMS
from treefarms.model.model_set import ModelSetContainer

df = pd.read_csv("../experiments/datasets/compas/binned.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]
h = df.columns[:-1]
print(df.head())

# train TREEFARMS model
config = {
    "regularization": 0.01,  # regularization penalizes the tree with more leaves. We recommend to set it to relative high value to find a sparse tree.
    "rashomon_bound_multiplier": 0.05,  # rashomon bound multiplier indicates how large of a Rashomon set would you like to get
}

model = TREEFARMS(config)

model.fit(X, y)

# model.visualize()

first_tree = model[0]
print(f'The accuracy of the first tree on the data is: {first_tree.score(X, y)}')
print(model[0])