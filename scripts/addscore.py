import numpy as np
import json
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
            description='add scores for annotations')
    parser.add_argument('anno')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fname = args.anno

    gt = json.load(open(fname))
    for i in range(len(gt['annotations'])):
        gt['annotations'][i]['score'] = 1.0
    
    json.dump(gt, open(fname.split('.')[0] + '_score.json','w'))

