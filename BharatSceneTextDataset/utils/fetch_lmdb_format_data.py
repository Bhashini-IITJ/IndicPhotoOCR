#!/usr/bin/env python3
"""a modified version of PARseq repository https://github.com/baudm/parseq/blob/main/tools/create_lmdb_dataset.py"""
import io
import os
import fire
import lmdb
import csv
import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    img = Image.open(io.BytesIO(imageBin)).convert('RGB')
    return np.prod(img.size) > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(recognition_folder_path, split, language, output_directory, checkValid=True):
    """
    Create LMDB dataset for a specific language from a specific split (train/test).
    ARGS:
        recognition_folder_path : path to the recognition folder
        split                   : dataset split to process ('train' or 'test')
        language                : specific language folder to process (e.g., 'english')
        output_directory        : LMDB output directory
        checkValid              : if true, check the validity of every image
    """
    inputPath = os.path.join(recognition_folder_path, split, language)
    gtFile = os.path.join(recognition_folder_path, f'{split}.csv')
    os.makedirs(output_directory, exist_ok=True)
    env = lmdb.open(output_directory, map_size=1099511627776)

    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        nSamples = 0
        for i, row in enumerate(reader):
            imagePath, label, lang = row
            if lang != language or not imagePath.startswith(f'{split}/{language}'):
                continue
            imagePath = os.path.join(recognition_folder_path, imagePath)
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
            if checkValid:
                try:
                    img = Image.open(io.BytesIO(imageBin)).convert('RGB')
                except IOError as e:
                    with open(output_directory + '/error_image_log.txt', 'a') as log:
                        log.write('{}-th image data occurred error: {}, {}\n'.format(i, imagePath, e))
                    continue
                if np.prod(img.size) == 0:
                    print('%s is not a valid image' % imagePath)
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
            nSamples += 1

    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)
