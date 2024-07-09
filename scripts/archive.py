# Copyright 2024, Bill Kennedy (https://github.com/rbbrdckybk/civitai-companion)
# SPDX-License-Identifier: MIT

import sys
import unicodedata
import re
import os
import shlex
import glob
from tqdm.auto import tqdm
from os.path import exists
from pathlib import Path
from datetime import datetime as dt
import scripts.utils as utils

# Handles archiving images
class Archive:
    def __init__(self, image_metadata, config):
        self.logfile = os.path.join('logs', 'log.txt')
        os.makedirs('logs', exist_ok = True)

        self.metadata = image_metadata

        self.archive_location = 'E:\Art\archive\Civitai'
        self.move_images = True
        self.rename_images = True
        self.organize_images = True
        self.log_to_console = False


    # performs selected archiving tasks
    def handle_archiving(self):
        if self.rename_images:
            self.handle_rename_images()


    # renames images
    def handle_rename_images(self):
        self.log('Renaming images...')

        # sort by base model
        models = []
        for k, v in self.metadata.items():
            if v.base_model.lower().strip() not in models:
                models.append(v.base_model.lower().strip())
        models.sort()

        # rename each file in format [date]_[base_model]_[#####].ext
        for m in models:
            model_count = 0
            for k, v in self.metadata.items():
                if v.base_model.lower().strip() == m:
                    orig_path = os.path.join(v.orig_filepath, v.orig_filename)
                    base = v.base_model.lower().strip()
                    if base == '':
                        base = 'unknown_base'

                    new_filename = dt.now().strftime('%Y-%m-%d') + '_' + base
                    ext = '.jpg'
                    if v.orig_filename.lower().endswith('.png'):
                        ext = '.png'

                    new_filename += '_' + str(model_count).zfill(5) + ext
                    new_path = os.path.join(v.orig_filepath, new_filename)

                    # try to rename, if error try to find another valid filename
                    try:
                        os.rename(orig_path, new_path)
                    except:
                        try:
                            new_filename = self.find_available_filename(v.orig_filepath, dt.now().strftime('%Y-%m-%d') + '_' + base, ext)
                            os.rename(orig_path, new_path)
                        except Exception as e:
                            self.log('Error: Unable to rename ' + v.orig_filename + ': ' + str(e))
                        else:
                            self.log('Renamed ' + v.orig_filename + ' to ' + new_filename + '...', self.log_to_console)
                            v.orig_filename = new_filename
                            model_count += 1
                    else:
                        self.log('Renamed ' + v.orig_filename + ' to ' + new_filename + '...', self.log_to_console)
                        v.orig_filename = new_filename
                        model_count += 1


    # finds a valid filename that isn't currently in-use
    def find_available_filename(self, path, filename, ext):
        count = 0
        fn = utils.sanitize_filename(filename) + '_' + str(count).zfill(5) + ext
        while exists(os.path.join(path, fn)):
            count += 1
            fn = utils.sanitize_filename(filename) + '_' + str(count).zfill(5) + ext
        return fn


    # re-orders the metadata so that models are grouped together
    def order_by_model(self):
        self.log('Ordering prompts by model...')
        models = []
        for k, v in self.metadata.items():
            if v.model.lower().strip() not in models:
                models.append(v.model.lower().strip())
        models.sort()

        # for each model,
        ordered_metadata = {}
        for m in models:
            for k, v in self.metadata.items():
                if v.model.lower().strip() == m:
                    ordered_metadata[k] = v

        #print('original count: ' + str(len(self.metadata.items())))
        #print('sorted count: ' + str(len(ordered_metadata.items())))
        self.metadata = ordered_metadata






    # handles logging to file/console
    def log(self, line, console = True):
        output = '[Archive] > ' + str(line)
        if console:
            print(output)
        with open(self.logfile, 'a', encoding="utf-8") as f:
            f.write(output + '\n')
