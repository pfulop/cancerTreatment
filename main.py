import os
import pandas as pd
import numpy as np

train_variant = pd.read_csv("./inputs/training_variants")
train_text = pd.read_csv("./inputs/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')

