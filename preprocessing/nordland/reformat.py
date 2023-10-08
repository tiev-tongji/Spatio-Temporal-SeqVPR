from scipy.io import loadmat
import numpy as np
from collections import namedtuple
import os
from os.path import join
import shutil
from tqdm import tqdm
dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr',
                                   'dbTimeStamp', 'qTimeStamp', 'gpsDb', 'gpsQ'])
def parse_db_struct(path):
    mat = loadmat(path)
    
    fieldnames = list(mat['dbStruct'][0, 0].dtype.names)

    dataset = mat['dbStruct'][0, 0]['dataset'].item()
    whichSet = mat['dbStruct'][0, 0]['whichSet'].item()

    dbImage = [f[0].item() for f in mat['dbStruct'][0, 0]['dbImageFns']]
    qImage = [f[0].item() for f in mat['dbStruct'][0, 0]['qImageFns']]

    numDb = mat['dbStruct'][0, 0]['numImages'].item()
    numQ = mat['dbStruct'][0, 0]['numQueries'].item()

    posDistThr = mat['dbStruct'][0, 0]['posDistThr'].item()
    posDistSqThr = mat['dbStruct'][0, 0]['posDistSqThr'].item()
    if 'nonTrivPosDistSqThr' in fieldnames:
        nonTrivPosDistSqThr = mat['dbStruct'][0, 0]['nonTrivPosDistSqThr'].item()
    else:
        nonTrivPosDistSqThr = None

    if 'dbTimeStamp' in fieldnames and 'qTimeStamp' in fieldnames:
        dbTimeStamp = [f[0].item() for f in mat['dbStruct'][0, 0]['dbTimeStamp'].T]
        qTimeStamp = [f[0].item() for f in mat['dbStruct'][0, 0]['qTimeStamp'].T]
        dbTimeStamp = np.array(dbTimeStamp)
        qTimeStamp = np.array(qTimeStamp)
    else:
        dbTimeStamp = None
        qTimeStamp = None

    if 'utmQ' in fieldnames and 'utmDb' in fieldnames:
        utmDb = mat['dbStruct'][0, 0]['utmDb'].T
        utmQ = mat['dbStruct'][0, 0]['utmQ'].T
    else:
        utmQ = None
        utmDb = None

    if 'gpsQ' in fieldnames and 'gpsDb' in fieldnames:
        gpsDb = mat['dbStruct'][0, 0]['gpsDb'].T
        gpsQ = mat['dbStruct'][0, 0]['gpsQ'].T
    else:
        gpsQ = None
        gpsDb = None

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr, dbTimeStamp, qTimeStamp, gpsQ, gpsDb)
    
def build(dbStruct, database, queries, path):
    dbImage = dbStruct.dbImage
    qImage = dbStruct.qImage
    northing = '0'
    extension = '.png'
    for i, img in tqdm(enumerate(dbImage)):
        season = database
        easting = dbStruct.utmDb[i].item()
        frame_num = dbStruct.utmDb[i].item()
        src_image_path = join(SR_DIR, season, img)
        img = img.strip('.png')
        rename = f"@{easting}@{northing}@{season}@{frame_num}@{img}@{extension}"
        dst_image_path = join(path, 'database', rename)        
        shutil.copy(src_image_path, dst_image_path)
        
    for i, img in tqdm(enumerate(qImage)):
        season = queries
        easting = dbStruct.utmQ[i].item()
        frame_num = dbStruct.utmQ[i].item()
        src_image_path = join(SR_DIR, season, img)
        img = img.strip('.png')
        rename = f"@{easting}@{northing}@{season}@{frame_num}@{img}@{extension}"
        dst_image_path = join(path, 'queries', rename)        
        shutil.copy(src_image_path, dst_image_path)
        

if __name__ == '__main__':
    db_path = '/media/autolab/seqNet/structFiles'
    # db_names = ['nordland_test_d-1_d2-1.db','nordland_train_d-40_d2-10.db','nordland_val_d-1_d2-1.db']
    db_names = ['nordland_val_d-1_d2-1.db']
    OUTPUT_DIR = 'nordland_formatted'
    SR_DIR = 'Nordland'
    train_val = ['summer','winter']
    test = ['spring','fall']
    for db_name in tqdm(db_names):
        split = db_name.split('_')[1]
        if split == 'test':
            database, queries = test
        else:
            database, queries = train_val
        path = join(OUTPUT_DIR, split)
        os.makedirs(path, exist_ok=True)
        os.makedirs(join(path,'queries'), exist_ok=True)
        dbStruct = parse_db_struct(join(db_path, db_name))
        build(dbStruct, database, queries, path)
    