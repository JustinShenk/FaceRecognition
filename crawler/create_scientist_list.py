import os
import json

ROOT_DIR = '../history_orig'

scientist = []

for subdir, dirs, files in os.walk(ROOT_DIR):
    for directory in dirs:
        scientist.append(directory.split('-')[:-1])

with open('scientist.json', 'w') as outfile:
    json.dump({'scientists': scientist}, outfile)
