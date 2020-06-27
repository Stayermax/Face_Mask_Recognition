import os
import argparse
import numpy as np
import pandas as pd
from report_copy import prediction_test

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

prediction_df = prediction_test(args.input_folder)
prediction_df.to_csv("prediction.csv", index=False, header=False)
