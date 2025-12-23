"""
This script will import data from various sources and place them inside the data folder.
"""

import os
import shutil

if not os.path.exists('data'):
    os.makedirs('data')

if os.path.exists('../geo_params_web/exports/c_min_k_max_params.csv'):
    shutil.copy('../geo_params_web/exports/c_min_k_max_params.csv', 'data/c_min_k_max_params.csv')
    print("Copied 'c_min_k_max_params.csv' to data folder.")
else:
    print("Warning: 'c_min_k_max_params.csv' not found in the source directory.")
    print("  You should run the ../geo_params_web/exports.py script to generate this file.")
