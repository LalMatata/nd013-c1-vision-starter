import argparse
import glob
import os
import random
import math
import numpy as np
import shutil
from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    ## data_dir arg using '/home/workspace/data/waymo'
    ## According to EDA, divide the data into 8:1:1 for train/val/test 
    source = os.path.join(data_dir,'training_and_validation')

    filePath = glob.glob(os.path.join(source,'*.tfrecord'))

    fileList = [file for file in filePath]
    np.random.shuffle(fileList)

    trainDes = os.path.join(data_dir,'train')
    valDes = os.path.join(data_dir,'val')
    testDes = os.path.join(data_dir,'test')
    
    #check folder exist
    os.makedirs(trainDes, exist_ok=True)
    os.makedirs(valDes, exist_ok=True)
    os.makedirs(testDes, exist_ok=True)
    
    ctn = len(fileList)
    print('Ctn',ctn)
    trainCtn = math.floor(0.8*ctn)
    valCtn = math.floor(0.9*ctn)
    
    for index, file in enumerate(fileList):
        path = os.path.join(data_dir,file)
        if index < trainCtn:
            shutil.move(path,trainDes)
        elif index < valCtn:
            shutil.move(path,valDes)
        else:
            shutil.move(path,testDes)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)