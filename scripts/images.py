# Copyright 2024, Bill Kennedy (https://github.com/rbbrdckybk/civitai-companion)
# SPDX-License-Identifier: MIT

import sys
import unicodedata
import re
import os
import shlex
import glob
import requests
import json
from tqdm.auto import tqdm
from os.path import exists
from pathlib import Path
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import scripts.utils as utils

# Handles examining a folder tree, finding civitai.com images and extracting their metadata
class Images:
    # config is a dict of options prepared by the Config class
    def __init__(self, config):
        self.logfile = os.path.join('logs', 'log.txt')
        os.makedirs('logs', exist_ok = True)

        # init civitai id/hash caches
        self.cache_id_file = os.path.join('cache', 'civitai_version_ids.txt')
        self.cache_hash_file = os.path.join('cache', 'civitai_hash_ids.txt')
        self.cache_id = {}
        self.cache_hash = {}
        os.makedirs('cache', exist_ok = True)
        self.init_caches()

        self.log_to_console = False
        self.image_path = config.get('path')
        self.ignore_subdirs = config.get('path_ignore_subdirs')
        self.metadata = {}

        # grab all images and decode any metadata
        self.images = self.collect_images()
        self.extract_metadata_from_images()
        self.decode_metadata()

        # do necessary lookups from cache and/or on civitai.com
        self.lookup_missing_metadata()

        # check that the type of resource specified in metadata matches what
        # the civitai.com API returns; use API if mismatched
        self.verify_resource_types()

        # attempt to infer base models from other metadata
        self.infer_base_models()


    # check each metadata resource type against API values
    # if mismatched, use API types
    # resource types listed in image metadata are sometimes incorrect
    def verify_resource_types(self):
        self.log('Verifying image metadata resource types match API values...')
        for k, v in self.metadata.items():
            for r in v.resources:
                api_type = self.lookup_civitai_resource_type(r.version_id)
                if api_type != '':
                    if r.type.lower().strip() == 'checkpoint':
                        r.type = 'model'
                    if r.type != api_type:
                        self.log('Warning: resource type (' + r.type + ') does not match API type (' + api_type + ') for resource ' + r.resource_name + '; using API type...', self.log_to_console)
                        r.type = api_type


    # returns a dict (version_id : filename) of referenced lora resources
    # types is a list of the types of resources to return (e.g. lora, embed, model)
    def get_referenced_resources(self, types):
        resources = {}
        # if a single resource was passed as a string instead of list, convert it
        if isinstance(types, str):
            t = types
            types = []
            types.append(t)
        # make sure everything is lowercase without any whitespace padding
        for t in types:
            t = t.lower().strip()

        for k, v in self.metadata.items():
            for r in v.resources:
                if r.type.lower().strip() in types and r.filename != '' and r.version_id != '':
                    resources[r.version_id] = r
        return resources


    # checks the local cache and then makes necessary API calls to civitai.com
    # to fill in any essential (model version ID) missing metadata
    def lookup_missing_metadata(self):
        if len(self.metadata) > 0:
            self.log('Querying civitai.com for missing info in images containing metadata...')
        for k, v in tqdm(self.metadata.copy().items()):
            md = v
            for r in v.resources:
                #if r.type in ('lora', 'embed') and r.filename == '':
                if r.filename == '':
                    # filename is missing; we need to look it up
                    if r.version_id == '' and r.hash != '':
                        # lookup the version ID with the hash info
                        vid = self.lookup_civitai_id(r.hash)
                        if vid != '':
                            # update metadata
                            index = 0
                            for o in md.resources:
                                if o.version_id == r.version_id:
                                    break
                                index += 1
                            md.resources[index].version_id = vid
                            self.metadata.update({k:md})
                            r.version_id = vid
                        else:
                            self.log('Unable to look up version ID for hash ' + r.hash + ' (' + v.orig_filename + ')!', self.log_to_console)

                    if r.version_id != '':
                        # lookup the name with the version ID
                        filename = self.lookup_civitai_filename(r.version_id)
                        if filename != '':
                            # update metadata
                            index = 0
                            for o in md.resources:
                                if o.version_id == r.version_id:
                                    break
                                index += 1
                            md.resources[index].filename = filename

                            # also fill in resource and base model names if present
                            md.resources[index].resource_name = self.lookup_civitai_resource_name(r.version_id)
                            md.resources[index].base_model = self.lookup_civitai_base_model(r.version_id)

                            self.metadata.update({k:md})
                        else:
                            self.log('Unable to look up filename for version ID ' + str(r.version_id) + ' (' + v.orig_filename + ')!', self.log_to_console)

    # given a civitai hash, looks up the version ID
    def lookup_civitai_id(self, hash):
        id = ''
        if hash in self.cache_hash:
            # check cache first
            id = self.cache_hash.get(hash)
        else:
            # lookup on civitai
            data = {}
            self.log('Looking up model hash ' + hash + ' on civitai.com...', self.log_to_console)
            info = requests.get('https://civitai.com/api/v1/model-versions/by-hash/' + str(hash))
            try:
                data = info.json()
            except:
                self.log('Error attempting to lookup model hash ' + hash + ' on civitai.com!')

            if 'id' in data:
                id = data['id']
                # add to cache
                self.write_cache_hash(hash, id)
            else:
                if 'error' in data:
                    error = data['error']
                    if error == 'Model not found':
                        self.log('Hash ' + hash + ' does not exist on civitai.com!', self.log_to_console)
                        self.write_cache_hash(hash, '')
        return id

    # given a civitai version ID, looks up the filename
    def lookup_civitai_filename(self, id):
        filename = ''
        if id in self.cache_id:
            # check cache first
            filename = self.cache_id.get(id)
            if ',' in filename:
                filename = filename.split(',', 1)[0]
        else:
            # lookup on civitai
            data = {}
            self.log('Looking up model version id ' + str(id) + ' on civitai.com...', self.log_to_console)
            info = requests.get('https://civitai.com/api/v1/model-versions/' + str(id))
            try:
                data = info.json()
                if 'files' in data:
                    for file in data['files']:
                        if 'downloadUrl' in file:
                            if file['downloadUrl'].endswith(str(id)):
                                if 'name' in file:
                                    filename = file['name']
                                    filename = utils.sanitize_filename(filename)
                                    # get additional details if present:
                                    name = ''
                                    if 'model' in data and 'name' in data['model']:
                                        name = data['model']['name']
                                    type = ''
                                    if 'model' in data and 'type' in data['model']:
                                        type = data['model']['type']
                                    base_model = ''
                                    if 'baseModel' in data:
                                        base_model = data['baseModel']
                                    # add to cache
                                    self.write_cache_id(id, filename, name, base_model, type)
                                    break
                else:
                    if 'error' in data:
                        error = data['error']
                        if error == 'Model not found':
                            self.log('Model version ID ' + str(id) + ' does not exist on civitai.com!', self.log_to_console)
                            self.write_cache_id(id, '', '', '', '')
            except:
                self.log('Error attempting to lookup model id ' + str(id) + ' on civitai.com!', self.log_to_console)

        return filename


    # given a civitai version ID, looks up the resource name
    # this only checks the cache (call this after filename lookups)
    def lookup_civitai_resource_name(self, id):
        name = ''
        if id in self.cache_id:
            temp = self.cache_id.get(id)
            if ',' in temp:
                data = temp.split(',')
                if len(data) >= 2:
                    name = data[1]
        return name


    # given a civitai version ID, looks up the base model
    # this only checks the cache (call this after filename lookups)
    def lookup_civitai_base_model(self, id):
        base = ''
        if id in self.cache_id:
            temp = self.cache_id.get(id)
            if ',' in temp:
                data = temp.split(',')
                if len(data) >= 3:
                    base = data[2]
        return base


    # given a civitai version ID, looks up the resource type
    # this only checks the cache (call this after filename lookups)
    def lookup_civitai_resource_type(self, id):
        type = ''
        if id in self.cache_id:
            temp = self.cache_id.get(id)
            if ',' in temp:
                data = temp.split(',')
                if len(data) >= 4:
                    type = data[3]
                    # translate type to match resource names in image metadata
                    # Semi-complete list? :
                    # LoCon, LORA, TextualInversion, Checkpoint, DoRA, VAE
                    if type.lower().strip() in ('lora', 'locon', 'dora'):
                        type = 'lora'
                    elif type.lower().strip() in ('textualinversion'):
                        type = 'embed'
                    elif type.lower().strip() in ('checkpoint', 'model'):
                        type = 'model'
                    elif type.lower().strip() in ('vae'):
                        type = 'vae'
        return type


    # writes a new civitai.com version ID/filename pair to the cache
    def write_cache_id(self, id, filename, resource_name, base_model, type):
        resource_name = resource_name.replace(',', ';')
        base_model = base_model.replace(',', ';')
        type = type.replace(',', ';')
        if id not in self.cache_id:
            self.cache_id[id] = filename + ',' + resource_name + ',' + base_model + ',' + type
            with open(self.cache_id_file, 'a', encoding="utf-8") as f:
                f.write(str(id) + ',' + filename + ',' + resource_name + ',' + base_model + ',' + type + '\n')

    # writes a new civitai.com hash/version ID pair to the cache
    def write_cache_hash(self, hash, id):
        if hash not in self.cache_hash:
            self.cache_hash[hash] = id
            with open(self.cache_hash_file, 'a', encoding="utf-8") as f:
                f.write(hash + ',' + str(id) + '\n')

    # read previously cached civitai.com version IDs/hash lookups into memory
    def init_caches(self):
        filepath = self.cache_id_file
        if exists(filepath):
            with open(filepath, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if ',' in line:
                    id = line.split(',', 1)[0].strip()
                    fn = line.split(',', 1)[1].strip()
                    if id != '' and id != '\n':
                        self.cache_id[id] = fn

        filepath = self.cache_hash_file
        if exists(filepath):
            with open(filepath, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if ',' in line:
                    id = line.split(',', 1)[0].strip()
                    fn = line.split(',', 1)[1].strip()
                    if id != '' and id != '\n':
                        self.cache_hash[id] = fn

    # for debugging, prints raw exif tags present in the given image
    def debug_print_metadata_info(self, image_path):
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            print('\nEXIF data present in ' + image_path + ':')
            for tag_id in exif_data:
                tag = TAGS.get(tag_id, tag_id)
                data = exif_data.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode(errors='replace')
                #print('  [' + str(tag_id) + '] : ' + str(tag) + ': ' + str(data))
                print('  [' + str(tag_id) + '] : ' + str(tag))
        else:
            print('\nNo EXIF data present in ' + image_path + '!')

    # for debugging; lists k/v metadata pairs
    def debug_list_metadata(self):
        self.log("Listing all extracted image metadata...:\n")
        for k, v in self.metadata.items():
            debug_str = 'Debugging info for ' + k + ':' + '\n'
            debug_str += 'Original Filename: ' + v.orig_filename + '\n'
            debug_str += 'Original Filepath: ' + v.orig_filepath + '\n'
            debug_str += 'Prompt: ' + v.prompt + '\n'
            debug_str += 'Negative Prompt: ' + v.neg_prompt + '\n'
            debug_str += 'Seed: ' + str(v.seed) + '\n'
            debug_str += 'Width: ' + str(v.width) + '\n'
            debug_str += 'Height: ' + str(v.height) + '\n'
            debug_str += 'Steps: ' + str(v.steps) + '\n'
            debug_str += 'Scale: ' + str(v.scale) + '\n'
            debug_str += 'Strength: ' + str(v.strength) + '\n'
            if v.base_model != '':
                debug_str += 'Model: ' + v.model + ' (base: ' + v.base_model + ')\n'
            else:
                debug_str += 'Model: ' + v.model + ' (unknown base model)\n'
            debug_str += 'Model Hash: ' + str(v.hash) + '\n'
            debug_str += 'Sampler: ' + v.sampler + '\n'
            debug_str += 'Clip Skip: ' + str(v.clip_skip) + '\n'
            debug_str += 'Resources: ' + self.debug_list_resources(v.resources)
            debug_str += 'Raw Metadata: ' + v.raw_metadata + '\n'
            self.log(debug_str)

    # for debugging; resources used in an image
    def debug_list_resources(self, res):
        resources = ''
        count = 0
        for r in res:
            count += 1
            resources += ' [' + str(count) + '] (' + r.type + ')\n'
            resources += '   Resource Name: ' + r.resource_name + '\n'
            resources += '   Version ID: ' + str(r.version_id) + '\n'
            resources += '   Hash: ' + r.hash + '\n'
            resources += '   Filename: ' + r.filename + '\n'
            resources += '   Base Model: ' + r.base_model + '\n'
            if r.type == 'lora':
                resources += '   LoRA Weight: ' + str(r.weight) + '\n'
        if resources == '':
            resources = 'none\n'
        else:
            resources = str(count) + ' total resources used:\n' + resources
        return resources


    # for debugging; lists the different types of resources found in metadata
    # types here are what the image metadata claims it is
    def debug_list_metadata_resource_types(self):
        output = "These types of resources were found in image metadata:\n"
        types = set()
        for k, v in self.metadata.items():
            for r in v.resources:
                types.add(r.type)
        for t in types:
            output += ' ' + t + '\n'
        self.log(output)


    # for debugging; lists the different types of resources found in metadata
    # types here are after looking up the version IDs of resources via API
    def debug_list_metadata_resource_types_via_api(self):
        output = "These types of resources were found in image metadata (API verified):\n"
        types = set()
        for k, v in self.metadata.items():
            for r in v.resources:
                api_type = self.lookup_civitai_resource_type(r.version_id)
                types.add(api_type)
        for t in types:
            output += ' ' + t + '\n'
        self.log(output)


    # for debugging; lists the different base models used in the images
    def debug_list_base_model_breakdown(self):
        output = "Base model breakdown by image count:\n"
        bases = set()
        counts = []
        d = {}
        for k, v in self.metadata.items():
            bases.add(v.base_model)
            counts.append(v.base_model)
        for base in bases:
            base = base.strip()
            if base == '':
                base = 'Unknown'
            count = counts.count(base)
            if count > 0:
                d[base] = count
        final = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for k, v in final:
            output += '  ' + k + ': ' + str(v) + '\n'
        self.log(output, self.log_to_console)


    # for debugging; lists the different main models used in the images
    def debug_list_model_breakdown(self, show_base=True):
        output = "Model breakdown by image count:\n"
        models = set()
        counts = []
        d = {}
        for k, v in self.metadata.items():
            m = v.model
            if v.hash != '':
                vid = self.lookup_civitai_id(v.hash)
                if vid != '':
                    filename = self.lookup_civitai_filename(vid)
                    if filename != '':
                        m = filename
            if m.endswith('.safetensors'):
                m = m[:-12]
            if show_base and v.base_model != '':
                m = m + ' (' + v.base_model + ')'
            models.add(m)
            counts.append(m)
        for model in models:
            model = model.strip()
            if model == '':
                model = 'Unknown'
            count = counts.count(model)
            if count > 0:
                d[model] = count
        final = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for k, v in final:
            output += '  ' + k + ': ' + str(v) + '\n'
        self.log(output, self.log_to_console)


    # for debugging; lists the different samplers used in the images
    def debug_list_sampler_breakdown(self):
        output = "Sampler breakdown by image count:\n"
        samplers = set()
        counts = []
        d = {}
        for k, v in self.metadata.items():
            s = v.sampler
            samplers.add(s)
            counts.append(s)
        for sampler in samplers:
            sampler = sampler.strip()
            if sampler == '':
                sampler = 'Unknown'
            count = counts.count(sampler)
            if count > 0:
                d[sampler] = count
        final = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for k, v in final:
            output += '  ' + k + ': ' + str(v) + '\n'
        self.log(output, self.log_to_console)
        return samplers


    # looks at all metadata and attempts to infer what base model was used for this image
    def infer_base_models(self):
        self.log("Attempting to infer base models for all images...")
        for k, v in tqdm(self.metadata.copy().items()):
            md = v
            base = ''

            if v.hash != '':
                # see if we have a cache entry for the main model hash
                vid = self.lookup_civitai_id(v.hash)
                if vid != '':
                    # the filename lookup is just to force a cache write if this is the
                    # first lookup of this version ID
                    filename = self.lookup_civitai_filename(vid)
                    if filename != '':
                        # if the above succeeded, this should return the proper base
                        base = self.lookup_civitai_base_model(vid)

            if base == '':
                # if we're not able to lookup directly via hash, we'll need to
                # make an educated guess based on other metadata
                # first examine resources for models
                bases = []
                for r in v.resources:
                    if r.type.lower() == 'checkpoint' and r.base_model != '':
                        bases.append(r.base_model)

                if len(bases) > 0:
                    if len(bases) > 1:
                        # multiple models in resource list
                        # this case does not seem to occur
                        pass
                    else:
                        # only one model found in the resources, assume it's the base
                        base = bases[0]

                # if we can't determine from metadata, anything with 'score_'
                # in the prompt should be for a pony model
                if base == '' and 'score_' in v.prompt:
                    base = 'Pony'

            if base != '':
                md.base_model = base
                self.metadata.update({k:md})

    # decodes relevent metadata present in all images
    def extract_metadata_from_images(self):
        if len(self.images) > 0:
            self.log('Looking for metadata in images...')
            for img in tqdm(self.images):
                self.log('Working on ' + img + '...', self.log_to_console)
                data = self.read_exif(img)
                if data == '':
                    self.log(img + ' contains no metadata!', self.log_to_console)
                    pass
                else:
                    md = ImageMetaData()
                    md.raw_metadata = data
                    md.orig_filename = os.path.basename(img)
                    dir = os.path.dirname(os.path.abspath(img))
                    md.orig_filepath = dir
                    self.metadata.update({img:md})

    # examines an image's EXIF data, returns the UserComment field if present
    def read_exif(self, image_path):
        if not exists(image_path):
            self.log('Error: ' + image_path + ' does not appear to exist (possible filename too long?)!')
            return ''
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is not None:
            for tag_id in exif_data:
                tag = TAGS.get(tag_id, tag_id)
                data = exif_data.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode(errors='replace')
                if tag == 'UserComment':
                    if data.startswith('UNICODE'):
                        data = data.replace('UNICODE', '', 1)
                    return data.replace('\x00', '')
        else:
            # try to extract comfy workflow
            prompt = ''
            try:
                metadata = image.text
                prompt = metadata["prompt"]
            except:
                prompt = ''
            return prompt
        return ''

    # collect all image files in the specified target location
    def collect_images(self):
        # build a list of images
        self.log('Collecting images in ' + self.image_path + '...')
        if not self.ignore_subdirs:
            images = self.get_images_from_tree(self.image_path)
            self.log('Found ' + str(len(images)) + ' images in ' + self.image_path + ' and all sub-directories...')
        else:
            images = self.get_images_from_dir(self.image_path)
            self.log('Found ' + str(len(images)) + ' images in ' + self.image_path + '...')
        return images

    # gets images found within specified dir, ignores subdirs
    def get_images_from_dir(self, dir):
        images = []
        for f in os.scandir(dir):
            if f.path.lower().endswith('.jpg') or f.path.lower().endswith('.jpeg') or f.path.lower().endswith('.png'):
                images.append(f.path)
        images.sort()
        return images

    # gets images found within specified dir and all sub-dirs
    def get_images_from_tree(self, root_dir):
        images = []
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                if name.lower().endswith('.jpg') or name.lower().endswith('.jpeg') or name.lower().endswith('.png'):
                    full_file_path = os.path.join(root, name)
                    images.append(full_file_path)

            for name in dirs:
                full_dir_path = os.path.join(root, name)
        images.sort()
        return images

    # handles logging to file/console
    def log(self, line, console = True):
        output = '[Images] > ' + str(line)
        if console:
            print(output)
        with open(self.logfile, 'a', encoding="utf-8") as f:
            f.write(output + '\n')

    # extracts SD parameters from the full command
    def decode_metadata(self):
        for key, val in self.metadata.copy().items():
            self.log('Decoding metadata for ' + val.orig_filename + '...', False)
            md = val
            command = md.raw_metadata
            if command != "":
                dream_factory = False
                is_json = True
                try:
                    json.loads(command)
                except:
                    is_json = False
                p = ''

                if is_json:
                    errors = 0
                    workflow = json.loads(command)
                    software = ''
                    is_comfy = True
                    if 'Fooocus v' in command:
                        software = 'Fooocus'
                        is_comfy = False
                        try:
                            md.prompt = utils.sanitize_prompt(workflow['prompt'])
                            md.neg_prompt = utils.sanitize_prompt(workflow['negative_prompt'])
                            md.steps = workflow['steps']
                            md.scale = workflow['guidance_scale']
                            res = workflow['resolution'].strip('(').strip(')')
                            md.width = res.split(',', 1)[0].strip()
                            md.height = res.split(',', 1)[1].strip()
                            md.sampler = workflow['sampler']
                            md.scheduler = workflow['scheduler']
                            md.seed = workflow['seed']
                            md.model = utils.extract_model_filename(workflow['base_model'])
                            md.hash = workflow['base_model_hash']
                        except:
                            errors += 1
                        else:
                            # if above succeeded, try to get loras as well
                            try:
                                loras = workflow['loras']
                                for lora in loras:
                                    rsc = ImageResources()
                                    rsc.type = 'lora'
                                    rsc.hash = lora[2]
                                    rsc.weight = lora[1]
                                    md.resources.append(rsc)
                            except:
                                pass

                    elif 'RuinedFooocus' in command:
                        # RuinedFooocus does not include LoRA hashes or civitai IDs so
                        # cannot look them up
                        software = 'RuinedFooocus'
                        is_comfy = False
                        try:
                            md.prompt = utils.sanitize_prompt(workflow['Prompt'])
                            md.neg_prompt = utils.sanitize_prompt(workflow['Negative'])
                            md.steps = workflow['steps']
                            md.scale = workflow['cfg']
                            md.width = workflow['width']
                            md.height = workflow['height']
                            md.sampler = workflow['sampler_name']
                            md.scheduler = workflow['scheduler']
                            md.seed = workflow['seed']
                            md.model = utils.extract_model_filename(workflow['base_model_name'])
                            md.hash = workflow['base_model_hash']
                        except:
                            errors += 1

                    if is_comfy:
                        # created by ComfyUI
                        # will not be 100% accurate for complex workflows with multiple prompts
                        software = 'ComfyUI'
                        for node in workflow:
                            data = workflow[node]
                            try:
                                if 'inputs' in data and 'text_positive' in data['inputs']:
                                    if isinstance(data['inputs']['text_positive'], str):
                                        md.prompt = utils.sanitize_prompt(data['inputs']['text_positive'].strip())
                                if 'inputs' in data and 'text_negative' in data['inputs']:
                                    if isinstance(data['inputs']['text_negative'], str):
                                        md.neg_prompt = utils.sanitize_prompt(data['inputs']['text_negative'].strip())
                                if 'inputs' in data and 'noise_seed' in data['inputs']:
                                    try:
                                        int(data['inputs']['noise_seed'])
                                    except:
                                        pass
                                    else:
                                        md.seed = data['inputs']['noise_seed']
                                if 'inputs' in data and 'sampler_name' in data['inputs']:
                                    if isinstance(data['inputs']['sampler_name'], str):
                                        md.sampler = data['inputs']['sampler_name']
                                if 'inputs' in data and 'scheduler' in data['inputs']:
                                    if isinstance(data['inputs']['scheduler'], str):
                                        md.scheduler = data['inputs']['scheduler']
                                    if 'steps' in data['inputs']:
                                        md.steps = data['inputs']['steps']
                                if 'inputs' in data and 'guidance' in data['inputs']:
                                    md.scale = data['inputs']['guidance']
                                if 'inputs' in data and 'unet_name' in data['inputs']:
                                    if isinstance(data['inputs']['unet_name'], str):
                                        md.model = utils.extract_model_filename(data['inputs']['unet_name'])
                                if 'inputs' in data and 'width' in data['inputs']:
                                    try:
                                        int(data['inputs']['width'])
                                    except:
                                        pass
                                    else:
                                        md.width = data['inputs']['width']
                                if 'inputs' in data and 'height' in data['inputs']:
                                    try:
                                        int(data['inputs']['height'])
                                    except:
                                        pass
                                    else:
                                        md.height = data['inputs']['height']
                                if 'inputs' in data and 'resolution' in data['inputs']:
                                    if isinstance(data['inputs']['resolution'], str):
                                        if 'x' in data['inputs']['resolution'].lower():
                                            md.width = data['inputs']['resolution'].lower().strip().split('x', 1)[0]
                                            md.height = data['inputs']['resolution'].lower().strip().split('x', 1)[1]
                                            if ' ' in md.height:
                                                md.height = md.height.split(' ', 1)[0]
                            except:
                                errors += 1
                        # second pass to look for prompt is other nodes if necessary
                        if md.prompt == '':
                            for node in workflow:
                                data = workflow[node]
                                try:
                                    if 'inputs' in data and 'text' in data['inputs']:
                                        if isinstance(data['inputs']['text'], str):
                                            md.prompt = utils.sanitize_prompt(data['inputs']['text'].strip())
                                except:
                                    errors += 1

                    else:
                        if software == '':
                            self.log('Unsupported JSON metadata format encountered: ' + val.orig_filename + '!', False)
                        else:
                            self.log('Unsupported JSON metadata format encountered (' + software + '): ' + val.orig_filename + '!', False)
                            
                    if errors > 0:
                        self.log('Error reading JSON metadata from ' + val.orig_filename + '!', False)

                else:
                    # not JSON
                    command = command.strip('"')
                    if '--neg_prompt' in command:
                        # this was created by Dream Factory
                        dream_factory = True
                        df_params = utils.extract_params_from_command(command)
                        md.prompt = utils.sanitize_prompt(df_params.get('prompt')).strip().strip('"')
                        md.neg_prompt = utils.sanitize_prompt(df_params.get('neg_prompt')).strip().strip('"')
                        md.steps = df_params.get('steps')
                        md.scale = df_params.get('scale')
                        md.strength = df_params.get('strength')
                        md.width = df_params.get('width')
                        md.height = df_params.get('height')
                        md.sampler = df_params.get('sampler')
                        md.seed = df_params.get('seed')
                        md.clip_skip = df_params.get('clip_skip')
                        md.model = utils.extract_model_filename(df_params.get('model'))
                        md.hash = utils.extract_model_hash(df_params.get('model'))

                    elif 'Negative prompt:' in command:
                        # we'll assume anything before this is the prompt
                        temp = command.split('Negative prompt:', 1)[0]
                        temp = temp.strip()
                        temp = temp.replace('\\', '')
                        md.prompt = utils.sanitize_prompt(temp)

                        # get negative
                        temp = command.split('Negative prompt:', 1)[1].strip()
                        if temp.startswith('Steps:'):
                            temp = ''
                        elif '\nSteps:' in temp:
                            temp = temp.split('\nSteps:', 1)[0]
                        elif '\n' in temp:
                            temp = temp.split('\n', 1)[0]

                        md.neg_prompt = utils.sanitize_prompt(temp).strip().strip('"')
                        p = command.split('Negative prompt:', 1)[1].strip()
                    else:
                        if '\n' in command:
                            temp = command.rsplit('\n', 1)[0].strip()
                            md.prompt = utils.sanitize_prompt(temp)
                            p = command.rsplit('\n', 1)[1]
                        else:
                            p = command

                    # get the rest of the params:
                    if not dream_factory:
                        if 'Steps:' in p and ',' in p:
                            v = p.split('Steps:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.steps = v

                        # TODO case-insensitive match/replace here
                        if 'CFG scale:' in p and ',' in p:
                            v = p.split('CFG scale:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.scale = v
                        elif 'CFG Scale:' in p and ',' in p:
                            v = p.split('CFG Scale:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.scale = v

                        if 'Denoising strength:' in p and ',' in p:
                            v = p.split('Denoising strength:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.strength = v

                        if 'Size:' in p and ',' in p:
                            v = p.split('Size:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            if 'x' in v:
                                width = v.split('x', 1)[0].strip()
                                height = v.split('x', 1)[1].strip()
                                md.width = width
                                md.height = height

                        if 'Clip skip:' in p and ',' in p:
                            v = p.split('Clip skip:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.clip_skip = v

                        if 'Sampler:' in p and ',' in p:
                            v = p.split('Sampler:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            if v.endswith(' Exponential'):
                                v = v.replace(' Exponential', '')
                            if v.endswith(' Karras'):
                                v = v.replace(' Karras', '')
                            md.sampler = v

                        if 'Seed:' in p and ',' in p:
                            v = p.split('Seed:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.seed = v

                        if 'Model:' in p and ',' in p:
                            v = p.split('Model:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.model = utils.extract_model_filename(v)

                        if 'Model hash:' in p and ',' in p:
                            v = p.split('Model hash:', 1)[1].strip()
                            v = v.split(',', 1)[0].strip()
                            md.hash = v

                    # get resources used:
                    if 'Civitai resources:' in p:
                        # option 1
                        # get loras
                        resources = p.split('Civitai resources:', 1)[1].strip()
                        while '{"type":"lora",' in resources and '}' in resources:
                            work = resources.split('{"type":"lora",', 1)[1].split('}', 1)[0]
                            if '"modelVersionId":' in work and ',' in work:
                                id = work.split('"modelVersionId":', 1)[1].split(',', 1)[0]
                                weight = 1.0
                                if '"weight":' in work and ',' in work:
                                    w = work.split('"weight":', 1)[1].split(',', 1)[0]
                                    try:
                                        weight = float(w)
                                    except:
                                        weight = 1.0
                                rsc = ImageResources()
                                rsc.type = 'lora'
                                rsc.version_id = id
                                rsc.weight = weight
                                md.resources.append(rsc)
                            before = resources.split('{"type":"lora",', 1)[0]
                            after = resources.split('{"type":"lora",', 1)[1].split('}', 1)[1]
                            resources = (before + after).strip()

                        # get checkpoints
                        resources = p.split('Civitai resources:', 1)[1].strip()
                        while '{"type":"checkpoint",' in resources and '}' in resources:
                            work = resources.split('{"type":"checkpoint",', 1)[1].split('}', 1)[0]
                            id = ''
                            if '"modelVersionId":' in work and ',' in work:
                                id = work.split('"modelVersionId":', 1)[1].split(',', 1)[0]
                            elif '"modelVersionId":' in work:
                                id = work.split('"modelVersionId":', 1)[1].strip()
                            if id != '':
                                rsc = ImageResources()
                                rsc.type = 'checkpoint'
                                rsc.version_id = id
                                md.resources.append(rsc)
                            before = resources.split('{"type":"checkpoint",', 1)[0]
                            after = resources.split('{"type":"checkpoint",', 1)[1].split('}', 1)[1]
                            resources = (before + after).strip()

                        # get embeddings
                        resources = p.split('Civitai resources:', 1)[1].strip()
                        while '{"type":"embed",' in resources and '}' in resources:
                            work = resources.split('{"type":"embed",', 1)[1].split('}', 1)[0]
                            if '"modelVersionId":' in work and ',' in work:
                                id = work.split('"modelVersionId":', 1)[1].split(',', 1)[0]
                                rsc = ImageResources()
                                rsc.type = 'embed'
                                rsc.version_id = id
                                md.resources.append(rsc)
                            before = resources.split('{"type":"embed",', 1)[0]
                            after = resources.split('{"type":"embed",', 1)[1].split('}', 1)[1]
                            resources = (before + after).strip()

                        # extra pass to get loras in different format
                        resources = p.split('Civitai resources:', 1)[1].strip()
                        while 'Type = lora }"' in resources and '}' in resources:
                            work = resources.split('Type = lora }"', 1)[1].split('}', 1)[0]
                            if '"modelVersionId":' in work:
                                id = work.split('"modelVersionId":', 1)[1]
                                found = True
                                try:
                                    num_id = int(id)
                                except:
                                    self.log('Unable to determine lora ID from metadata in ' + md.orig_filename, self.log_to_console)
                                    found = False
                                if found:
                                    weight = 1.0
                                    if '"weight":' in work and ',' in work:
                                        w = work.split('"weight":', 1)[1].split(',', 1)[0]
                                        try:
                                            weight = float(w)
                                        except:
                                            weight = 1.0
                                    rsc = ImageResources()
                                    rsc.type = 'lora'
                                    rsc.version_id = id
                                    rsc.weight = weight
                                    md.resources.append(rsc)
                            before = resources.split('Type = lora }"', 1)[0]
                            after = resources.split('Type = lora }"', 1)[1].split('}', 1)[1]
                            resources = (before + after).strip()

                    elif 'Hashes: {' in p:
                        # option 2
                        resources = p.split('Hashes: {', 1)[1].strip()
                        resources = resources.split('}', 1)[0].strip()
                        while ':' in resources:
                            while resources.startswith(' ') or resources.startswith('\"'):
                                resources = resources[1:]

                            type = resources.split('\"', 1)[0].strip('\"').lower()
                            if ':' in type:
                                type = type.split(':', 1)[0].strip()
                            resources = resources.split('\"', 1)[1]

                            while resources.startswith(' ') or resources.startswith(':')  or resources.startswith('\"'):
                                resources = resources[1:]

                            # handles case where we have something like this in metadata:  Hashes: {"model": ""}
                            if resources.strip() != '':
                                hash = resources.split('\"', 1)[0].strip()
                                resources = resources.split('\"', 1)[1]

                                while resources.startswith(' ') or resources.startswith(','):
                                    resources = resources[1:]

                                rsc = ImageResources()
                                rsc.type = type
                                rsc.hash = hash
                                md.resources.append(rsc)

                    elif 'Lora hashes: "' in p:
                        # option 3
                        resources = p.split('Lora hashes: \"', 1)[1].strip()
                        resources = resources.split('\"', 1)[0].strip()

                        while ':' in resources:
                            resources = resources.split(':', 1)[1]
                            hash = ''
                            if ',' in resources:
                                hash = resources.split(',', 1)[0].strip()
                                resources = resources.split(',', 1)[1]
                            else:
                                hash = resources.strip()

                            rsc = ImageResources()
                            rsc.type = 'lora'
                            rsc.hash = hash
                            md.resources.append(rsc)

            # save orig raw versions of prompt/neg prompt
            md.prompt_raw = md.prompt
            md.neg_prompt_raw = md.neg_prompt

            self.metadata.update({key:md})


# for organizing image metadata
class ImageMetaData:
  def __init__(self):
    self.orig_filename = ''
    self.orig_filepath = ''
    self.raw_metadata = ''
    self.prompt = ''
    self.prompt_raw = ''
    self.neg_prompt = ''
    self.neg_prompt_raw = ''
    self.seed = ''
    self.width = ''
    self.height = ''
    self.steps = ''
    self.scale = ''
    self.strength = ''
    self.model = ''
    self.hash = ''
    self.base_model = ''
    self.sampler = ''
    self.scheduler = ''
    self.clip_skip = ''
    self.resources = []

# for organizing resources
class ImageResources:
  def __init__(self):
    self.type = ''
    self.version_id = ''
    self.hash = ''
    self.filename = ''
    self.resource_name = ''
    self.base_model = ''
    self.weight = 1.0
