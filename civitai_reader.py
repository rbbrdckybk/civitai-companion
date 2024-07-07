# Copyright 2021 - 2024, Bill Kennedy (https://github.com/rbbrdckybk/dream-factory)
# SPDX-License-Identifier: MIT

import sys
import os
import glob
import argparse
from os.path import exists
from pathlib import Path
from scripts.images import Images
from scripts.network import Network
from scripts.prompts import Prompts
from scripts.config import Config

# gets resources found within specified dir and all sub-dirs
def get_resources_from_tree(root_dir):
    resources = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            if (name.lower().endswith('.safetensors') or
                name.lower().endswith('.ckpt') or
                name.lower().endswith('.pt')):
                full_file_path = os.path.join(root, name)
                resources.append(name)
        for name in dirs:
            full_dir_path = os.path.join(root, name)
    resources = list(set(resources))
    resources.sort()
    return resources

# handles creation/backup of logfile
def create_logfile(file):
    logfile = file
    os.makedirs('logs', exist_ok = True)
    if exists(logfile + '.bak'):
        os.remove(logfile + '.bak')
    if exists(logfile):
        os.rename(logfile, logfile + '.bak')

# logs to file/console
def log(file, line, console = True):
    logfile = file
    if console:
        print(line)
    output = '[Main] > ' + str(line)
    with open(logfile, 'a', encoding="utf-8") as f:
        f.write(output.replace('\n', '') + '\n')

# downloads missing resources
# missing_dict: dictionary of missing items (key = version id, val = images.ImageResources)
# descriptor: plural text to display in log describing what we're downloading (e.g. loras, embeddings, etc)
# download_location: where to save downloaded files
def download_resources(missing_dict, descriptor, download_location):
    if download_location != '':
        if len(missing_dict) > 0:
            log(logfile, '\nDownloading missing ' + descriptor + '...')
        count = 0
        for k, v in missing_dict.items():
            count += 1
            log(logfile, '  [' + str(count) + ' of ' + str(len(missing_dict)) + '] Attempting to download ' + v.filename + '...')
            network.download_file('https://civitai.com/api/download/models/' + str(k), download_location, v.filename)
    else:
        log(logfile, '\nDownload location not specified for ' + descriptor + '; no resources of this type will be downloaded!')

# entry point
if __name__ == '__main__':
    logfile = os.path.join('logs', 'log.txt')
    create_logfile(logfile)
    log(logfile, '\nStarting..\n')

    config = Config(argparse.ArgumentParser())
    #config.debug_display_user_options()

    if config.image_config.get('path') == '':
        print('Error: no image path specified; aborting!')
        print('(Start with --help for help; must specify image path on command line or via config file)')
        exit(0)

    network = Network(config.network_config)
    images = Images(config.image_config)

    #images.debug_list_metadata()
    #images.debug_list_metadata_resource_types()
    images.debug_list_base_model_breakdown()
    images.debug_list_model_breakdown(True)
    #images.debug_list_sampler_breakdown()

    print('')
    prompts = Prompts(images.metadata, config.prompt_config)
    prompts.manifest()
    prompts.write_prompt_file()

    log(logfile, '\nComparing referenced resources to existing local resources...')
    existing_loras = get_resources_from_tree(config.general_config.get('existing_lora_path'))
    existing_embeds = get_resources_from_tree(config.general_config.get('existing_embedding_path'))
    existing_checkpoints = get_resources_from_tree(config.general_config.get('existing_model_path'))

    referenced_loras = prompts.get_referenced_resources('lora')
    referenced_embeds = prompts.get_referenced_resources('embed')
    referenced_checkpoints = prompts.get_referenced_resources(['model', 'checkpoint'])

    missing_loras = {}
    for k, v in referenced_loras.items():
        if v.filename not in existing_loras:
            if k not in network.do_not_download:
                missing_loras[k] = v

    missing_embeds = {}
    for k, v in referenced_embeds.items():
        if v.filename not in existing_embeds:
            if k not in network.do_not_download:
                missing_embeds[k] = v

    missing_checkpoints = {}
    for k, v in referenced_checkpoints.items():
        if v.filename not in existing_checkpoints:
            if k not in network.do_not_download:
                missing_checkpoints[k] = v

    log(logfile, '  ' + str(len(missing_embeds)) + ' referenced embeddings(s) don\'t exist locally and need to be downloaded.')
    log(logfile, '  ' + str(len(missing_loras)) + ' referenced LoRA(s) don\'t exist locally and need to be downloaded.')
    log(logfile, '  ' + str(len(missing_checkpoints)) + ' referenced model(s) don\'t exist locally and need to be downloaded.')

    download_resources(missing_embeds, 'embeddings', config.general_config.get('download_embedding_path'))
    download_resources(missing_loras, 'LoRAs', config.general_config.get('download_lora_path'))
    download_resources(missing_checkpoints, 'models', config.general_config.get('download_model_path'))

    log(logfile, '\nDone!')
