# Copyright 2024, Bill Kennedy (https://github.com/rbbrdckybk/civitai-companion)
# SPDX-License-Identifier: MIT

import sys
import os
import glob
from os.path import exists
from pathlib import Path
from datetime import datetime as dt
import scripts.utils as utils
from scripts.images import ImageMetaData, ImageResources

# Handles outputting found image metadata as DF .prompts files
class Prompts:
    # config is a dict of options prepared by the Config class
    def __init__(self, image_metadata, config):
        self.logfile = os.path.join('logs', 'log.txt')
        os.makedirs('logs', exist_ok = True)

        self.metadata = image_metadata
        self.prepend_filename = config.get('append_filename')

        self.min_steps = config.get('min_steps')
        self.max_steps = config.get('max_steps')
        self.min_scale = config.get('min_scale')
        self.max_scale = config.get('max_scale')
        self.fix_resolution = config.get('fix_resolution')

        # set up list of base models to include
        self.base_list = []
        if config.get('only_include_base') != '':
            self.base_list = config.get('only_include_base').split(',')
            temp = []
            for b in self.base_list.copy():
                b = b.lower().strip()
                temp.append(b)
            self.base_list = temp

        # set up list of words to exclude from prompts if found
        self.word_filter_list = []
        if config.get('word_filter_list') != '':
            self.word_filter_list = config.get('word_filter_list').split(',')
            temp = []
            for w in self.word_filter_list.copy():
                w = w.lower().strip()
                temp.append(w)
            self.word_filter_list = temp

        # set up list of words to exclude from negative prompts if found
        self.neg_word_filter_list = []
        if config.get('neg_word_filter_list') != '':
            self.neg_word_filter_list = config.get('neg_word_filter_list').split(',')
            temp = []
            for w in self.neg_word_filter_list.copy():
                w = w.lower().strip()
                temp.append(w)
            self.neg_word_filter_list = temp

        # set up list of loras to exclude from prompts if found
        self.lora_filter_list = []
        if config.get('lora_filter_list') != '':
            self.lora_filter_list = config.get('lora_filter_list').split(',')
            temp = []
            for l in self.lora_filter_list.copy():
                l = l.lower().strip()
                temp.append(l)
            self.lora_filter_list = temp

        self.output_template = config.get('output_template')
        self.output_header = config.get('output_header')
        self.output_footer = config.get('output_footer')

        self.output_save_as = config.get('output_save_as')
        self.output_save_as = utils.ireplace('[date]', dt.now().strftime('%Y-%m-%d'), self.output_save_as)
        self.output_save_as = utils.ireplace('[time]', dt.now().strftime('%H-%M-%S'), self.output_save_as)

        # supported Auto1111 samplers as of 2024-07-02
        self.auto1111_samplers = ['DDIM',
            'DPM adaptive',
            'DPM fast',
            'DPM++ 2M',
            'DPM++ 2M SDE',
            'DPM++ 2M SDE Heun',
            'DPM++ 2S a',
            'DPM++ 3M SDE',
            'DPM++ SDE',
            'DPM2',
            'DPM2 a',
            'Euler',
            'Euler a',
            'Heun',
            'LCM',
            'LMS',
            'PLMS',
            'Restart',
            'UniPC'
        ]


    # standard/suggested method of running through task manifest
    # will create final clean prompts that are ready for write_prompt_file()
    def manifest(self):
        self.filter_unwanted_base_prompts()
        self.fix_lora_refs()
        self.add_missing_lora_refs()
        self.enforce_limits()
        self.check_samplers()
        self.remove_filter_words()
        self.remove_neg_filter_words()
        self.remove_empty(5)
        self.remove_dupes()
        self.remove_filter_loras()


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


    # removes filter words if they appear in prompts
    def check_samplers(self):
        self.log('Checking prompts for unsupported samplers:')
        count = 0
        for k, v in self.metadata.items():
            sampler = self.verify_sampler(v.sampler)
            if sampler != v.sampler:
                v.sampler = sampler
                count += 1
        self.log('Substituted ' + str(count) + ' unsupported sampler name(s) for their approximate Auto1111 equivalents...')


    # removes filter words if they appear in prompts
    def remove_filter_words(self):
        if len(self.word_filter_list) > 0:
            self.log('Checking prompts for filter words:')
            count = 0
            for k, v in self.metadata.items():
                for fw in self.word_filter_list:
                    if fw in v.prompt.lower():
                        before = len(v.neg_prompt)
                        # try to remove the filter word
                        v.prompt = utils.word_replace(fw, v.prompt)
                        # re-sanitize
                        v.prompt = utils.sanitize_prompt(v.prompt)
                        if before != len(v.neg_prompt):
                            count += 1
            self.log('Removed ' + str(count) + ' occurances of filter word(s) in prompts...')


    # removes filter words if they appear in negative prompts
    def remove_neg_filter_words(self):
        if len(self.neg_word_filter_list) > 0:
            self.log('Checking negative prompts for filter words:')
            count = 0
            for k, v in self.metadata.items():
                for fw in self.neg_word_filter_list:
                    if fw in v.neg_prompt.lower():
                        before = len(v.neg_prompt)
                        # try to remove the filter word
                        v.neg_prompt = utils.word_replace(fw, v.neg_prompt)
                        # re-sanitize
                        v.neg_prompt = utils.sanitize_prompt(v.neg_prompt)
                        if before != len(v.neg_prompt):
                            count += 1
            self.log('Removed ' + str(count) + ' occurances of filter word(s) in negative prompts...')


    # removes unwanted loras from prompts if they appear in the filter list
    def remove_filter_loras(self):
        if len(self.lora_filter_list) > 0:
            self.log('Checking prompts for unwanted LoRA references:')
            count = 0
            for k, v in self.metadata.items():
                temp = v.prompt
                while '<lora:' in temp and '>' in temp:
                    # save portions of prompt that are before/after lora ref
                    before = temp.split('<lora:', 1)[0]
                    after = temp.split('<lora:', 1)[1].split('>', 1)[1]
                    # this is inside a lora declaration
                    work = temp.split('<lora:', 1)[1].split('>', 1)[0]
                    work = work.split(':', 1)[0]
                    for fl in self.lora_filter_list:
                        if fl == work.lower().strip():
                            # remove this lora
                            temp = before + after
                            count += 1
                        else:
                            # this lora is fine; keep it
                            temp = before + '<zzzora:' + work + '>' + after
                v.prompt = temp.replace('<zzzora:', '<lora:')
            self.log('Removed ' + str(count) + ' occurances of unwanted LoRA(s) in prompts...')


    # adjusts metadata parameters so values are not outside of user-specified limits
    def enforce_limits(self):
        self.log('Checking prompt parameters for values outside user-specified limits:')
        changes = 0
        for k, v in self.metadata.items():
            if v.steps != '' and int(self.min_steps) > 0 and int(v.steps) < int(self.min_steps):
                v.steps = str(self.min_steps)
                changes += 1
            if v.steps != '' and int(self.max_steps) > 0 and int(v.steps) > int(self.max_steps):
                v.steps = str(self.max_steps)
                changes += 1
            if v.scale != '' and float(self.min_scale) > 0 and float(v.scale) < float(self.min_scale):
                v.scale = str(self.min_scale)
                changes += 1
            if v.scale != '' and float(self.max_scale) > 0 and float(v.scale) > float(self.max_scale):
                v.scale = str(self.max_scale)
                changes += 1
            if self.fix_resolution:
                new_res = self.fix_image_resolution(v.width, v.height, v.base_model)
                if new_res[0] != v.width:
                    v.width = str(new_res[0])
                    v.height = str(new_res[1])
                    changes += 1
        self.log('Made ' + str(changes) + ' adjustment(s) to prompt parameters...')


    # given a resolution, returns a new resolution closest to an official supported base resolution
    # for the specified platform
    def fix_image_resolution(self, width, height, platform='sdxl 1.0'):
        res = [width, height]
        try:
            int(width)
            int(height)
        except:
            #print('Error: width/height are not integers!')
            return res

        w = int(width)
        h = int(height)

        # check for invalid/zero resolution
        if w == 0 or h == 0:
            return res

        # get aspect ratio of user-supplied resolution
        width_larger = True
        aspect_ratio = w / h
        if h > w:
            width_larger = False
            aspect_ratio = h / w

        # find the closest official supported a/r
        if platform.lower().strip().startswith('sd 1.5'):
            # SD 1.5 supported/popular resolutions:
            # 512x512, 640x512, 768x512, 896x512
            supported_ar = [1.0000, 1.2500, 1.5000, 1.7500]
            closest_ar = min(supported_ar, key=lambda x:abs(x-aspect_ratio))
            if closest_ar == supported_ar[0]:
                # people generate square SD 1.5 images at various sizes
                square_sizes = [512, 640, 768]
                closest_size = min(square_sizes, key=lambda x:abs(x-w))
                res = [closest_size, closest_size]
            elif closest_ar == supported_ar[1]:
                res = [640, 512]
            elif closest_ar == supported_ar[2]:
                res = [768, 512]
            elif closest_ar == supported_ar[3]:
                res = [896, 512]

        elif platform.lower().strip().startswith('sd 2.1'):
            # SD 2.0/2.1 supported/popular resolutions:
            # 768x768, 896x768, 1024x768, 1152x768, 1280x768
            supported_ar = [1.0000, 1.1667, 1.3333, 1.5000, 1.6667]
            closest_ar = min(supported_ar, key=lambda x:abs(x-aspect_ratio))
            if closest_ar == supported_ar[0]:
                res = [768, 768]
            elif closest_ar == supported_ar[1]:
                res = [896, 768]
            elif closest_ar == supported_ar[2]:
                res = [1024, 768]
            elif closest_ar == supported_ar[3]:
                res = [1152, 768]
            elif closest_ar == supported_ar[4]:
                res = [1280, 768]

        else:
            # assume SDXL or derivitive, supported resolutions:
            # 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640
            supported_ar = [1.0000, 1.2857, 1.4615, 1.7500, 2.4000]
            closest_ar = min(supported_ar, key=lambda x:abs(x-aspect_ratio))
            if closest_ar == supported_ar[0]:
                res = [1024, 1024]
            elif closest_ar == supported_ar[1]:
                res = [1152, 896]
            elif closest_ar == supported_ar[2]:
                res = [1216, 832]
            elif closest_ar == supported_ar[3]:
                res = [1344, 768]
            elif closest_ar == supported_ar[4]:
                res = [1536, 640]

        # flip the width/height if the height was the larger original dimension
        if not width_larger:
            res.reverse()

        #print('\nPlatform: ' + platform)
        #print('User resolution: ' + width + 'x' + height)
        #print('User aspect ratio: ' + str(aspect_ratio))
        #print('Closest official ratio: ' + str(closest_ar))
        #print('Final resolution: ' + str(res[0]) + 'x' + str(res[1]))
        return res


    # removes prompts that don't match user-specified base models
    def filter_unwanted_base_prompts(self):
        if len(self.base_list) > 0:
            self.log('Filtering out prompts that don''t match these base models: ' + str(self.base_list) + '...')
            unwanted_keys = []
            original_length = len(self.metadata)
            for kc, vc in self.metadata.copy().items():
                for k, v in self.metadata.items():
                    if v.base_model.lower().strip() not in self.base_list:
                        # this is unwanted, save the key
                        unwanted_keys.append(k)
            for key in unwanted_keys:
                # remove all saved keys
                self.metadata.pop(key, None)
            final_length = len(self.metadata)
            num_unwanted = original_length - final_length
            self.log('Removed ' + str(num_unwanted) + ' unwanted prompt(s)...')


    # writes an output .prompts file containing prompts discerned from image data
    # in the format specified by the user-supplied template file
    def write_prompt_file(self):
        if len(self.metadata) == 0:
            self.log('No usable metadata to output; aborting prompt file write!')
            return
        filename = 'output.prompts'
        if self.output_save_as != '':
            # create output directory if necessary
            dir = os.path.dirname(os.path.abspath(self.output_save_as))
            os.makedirs(dir, exist_ok = True)
            filename = self.output_save_as
        use_default = True
        if self.output_template != '':
            if exists(self.output_template):
                use_default = False
            else:
                self.log('Error: specified prompt template file does not exist (' + self.output_template + '); using default!')
        if use_default:
            self.write_default_prompt_file(filename)
        else:
            template = ''
            with open(self.output_template, encoding = 'utf-8') as t:
                template = t.read()
            self.log('Writing prompt metadata to disk using supplied template (' + self.output_template + '):')
            f = open(filename, 'w', encoding = 'utf-8')
            f.write('#######################################################################################################\n')
            f.write('# ' + str(len(self.metadata)) + ' unique prompts from metadata extracted from civitai.com images.\n')
            f.write('# Created on ' + dt.now().strftime('%Y-%m-%d') + ' at ' + dt.now().strftime('%H:%M:%S') + '.\n')
            f.write('#######################################################################################################\n')
            count = 0
            for k, v in self.metadata.items():
                count += 1
                t = template
                ref = str(count).zfill(5)
                # handle template replacements
                t = utils.ireplace('[ref_num]', ref, t)
                t = utils.ireplace('[filename]', v.orig_filename, t)
                t = utils.ireplace('[filepath]', os.path.join(v.orig_filepath, v.orig_filename), t)
                t = utils.ireplace('[raw_metadata]', v.raw_metadata.replace('\n', '\n#'), t)
                t = utils.ireplace('[model]', v.model, t)
                t = utils.ireplace('[seed]', str(v.seed), t)
                t = utils.ireplace('[sampler]', v.sampler, t)
                t = utils.ireplace('[clip_skip]', str(v.clip_skip), t)
                t = utils.ireplace('[width]', str(v.width), t)
                t = utils.ireplace('[height]', str(v.height), t)
                t = utils.ireplace('[steps]', str(v.steps), t)
                t = utils.ireplace('[scale]', str(v.scale), t)
                t = utils.ireplace('[neg_prompt]', v.neg_prompt, t)
                t = utils.ireplace('[neg_prompt_raw]', v.neg_prompt_raw, t)
                t = utils.ireplace('[prompt]', v.prompt, t)
                t = utils.ireplace('[prompt_raw]', v.prompt_raw, t)
                t = utils.ireplace('[base_model]', v.base_model, t)
                t = utils.ireplace('[model_hash]', v.hash, t)
                # write templated prompt to output file
                f.write('\n' + t + '\n')
            f.close()
            self.log(str(len(self.metadata)) + ' prompts saved as ' + filename + '!')
        # add header/footer to output if they were specified
        self.attach_header_footer(filename)


    # writes a default Dream Factory .prompts file containing prompts discerned from image data
    def write_default_prompt_file(self, filename):
        self.log('Writing prompt metadata to disk using default template:')
        f = open(filename, 'w', encoding = 'utf-8')
        f.write('#######################################################################################################\n')
        f.write('# ' + str(len(self.metadata)) + ' unique prompts from metadata extracted from civitai.com images.\n')
        f.write('# Created on ' + dt.now().strftime('%Y-%m-%d') + ' at ' + dt.now().strftime('%H:%M:%S') + '.\n')
        f.write('#######################################################################################################\n')
        count = 0
        for k, v in self.metadata.items():
            count += 1
            f.write('\n')
            f.write('#######################################################################################################\n')
            f.write('# PROMPT ' + str(count).zfill(5) + '\n')
            f.write('# Extracted from: ' + v.orig_filename + '\n')
            f.write('# Raw metadata below:\n#' + v.raw_metadata.replace('\n', '\n#') + '\n\n')
            f.write('#######################################################################################################\n\n')
            f.write('!FILENAME = ' + str(count).zfill(5) + '-' + self.prepend_filename + '\n')
            f.write('#!CKPT_FILE = ' + v.model + '\n')
            f.write('#!SEED = ' + str(v.seed) + '\n')
            f.write('#!SAMPLER = ' + v.sampler + '\n')
            f.write('#!CLIP_SKIP = ' + str(v.clip_skip) + '\n')
            f.write('#!WIDTH = ' + str(v.width) + '\n')
            f.write('#!HEIGHT = ' + str(v.height) + '\n')
            f.write('!STEPS = ' + str(v.steps) + '\n')
            f.write('!SCALE = ' + str(v.scale) + '\n')
            f.write('\n!NEG_PROMPT = ' + v.neg_prompt + '\n')
            f.write('\n' + v.prompt + '\n')
        f.close()
        self.log(str(len(self.metadata)) + ' prompts saved as ' + filename + '!')


    # attaches header & footer to given file, if specified
    def attach_header_footer(self, filename):
        header = ''
        footer = ''
        if self.output_header != '':
            if exists(self.output_header):
                with open(self.output_header, encoding = 'utf-8') as f:
                    header = f.read()
            else:
                self.log('Error: specified output header file does not exist (' + self.output_header + '); ignoring it!')
        if self.output_footer != '':
            if exists(self.output_footer):
                with open(self.output_footer, encoding = 'utf-8') as f:
                    footer = f.read()
            else:
                self.log('Error: specified output footer file does not exist (' + self.output_footer + '); ignoring it!')

        # if we have a valid non-empty header or footer, attach it to the given file
        if header != '' or footer != '':
            prompts = ''
            if exists(filename):
                # save the original file content
                with open(filename, encoding = 'utf-8') as f:
                    prompts = f.read()

            # write the header, original content, and footer back to the file
            with open(filename, 'w', encoding = 'utf-8') as f:
                f.write(header)
                f.write(prompts)
                f.write(footer)


    # remove prompts that have a length under char_limit
    # lora references do not count against the limit
    def remove_empty(self, char_limit = 1):
        self.log('Removing prompts with less than ' + str(char_limit) + ' character(s):')
        empty_keys = []
        original_length = len(self.metadata)

        for k, v in self.metadata.items():
            prompt = v.prompt.lower()
            # remove lora references
            while '<lora:' in prompt and '>' in prompt:
                before = prompt.split('<lora:', 1)[0]
                start_ref = prompt.split('<lora:', 1)[1]
                after = start_ref.split('>', 1)[1]
                prompt = before + after
            if len(prompt.strip()) < char_limit:
                # this prompt is too short, save the key
                empty_keys.append(k)

        for key in empty_keys:
            # remove all saved keys
            self.metadata.pop(key, None)
        final_length = len(self.metadata)
        num_empty = original_length - final_length
        self.log('Removed ' + str(num_empty) + ' prompt(s) that were too short...')


    # checks metadata for duplicate prompts and removes them
    # a prompt is considered a duplicate if the prompt/negative prompt pair
    # is identical to another after converting both to lowercase
    def remove_dupes(self):
        self.log('Checking prompts for duplicates:')
        dupe_keys = []
        original_length = len(self.metadata)
        for kc, vc in self.metadata.copy().items():
            for k, v in self.metadata.items():
                if k != kc:
                    if v.prompt.lower() == vc.prompt.lower() and v.neg_prompt.lower() == vc.neg_prompt.lower():
                        # this is a dupe, save the key
                        dupe_keys.append(k)
        for key in dupe_keys:
            # remove all saved keys
            self.metadata.pop(key, None)
        final_length = len(self.metadata)
        num_dupes = original_length - final_length
        self.log('Removed ' + str(num_dupes) + ' duplicate prompt(s)...')


    # adds lora references that appears in resources metadata
    # but are absent from the actual prompt
    def add_missing_lora_refs(self):
        self.log('Checking prompts for missing lora references:')
        additions = 0
        for k, v in self.metadata.items():
            for r in v.resources:
                if 'lora' in r.type and r.filename != '':
                    # this reference is missing; add it
                    lref = '<lora:' + r.filename.rsplit('.', 1)[0]
                    if lref.lower() not in v.prompt.lower():
                        v.prompt += ' ' + lref + ':' + str(r.weight) + '>'
                        additions += 1
        self.log('Added ' + str(additions) + ' lora reference(s) that were missing from prompts...')


    # removes path references in prompt lora declarations
    def fix_lora_refs(self):
        self.log('Examining prompts for lora path references:')
        replacements = 0
        for k, v in self.metadata.items():
            temp = v.prompt
            while '<lora:' in temp and '>' in temp:
                # this is inside a lora declaration
                work = temp.split('<lora:', 1)[1].split('>', 1)[0]
                # keep only the filename, remove paths
                if '\\' in work:
                    work = work.rsplit('\\', 1)[1]
                    replacements += 1
                if '/' in work:
                    work = work.rsplit('/', 1)[1]
                    replacements += 1
                before = temp.split('<lora:', 1)[0]
                after = temp.split('<lora:', 1)[1].split('>', 1)[1]
                temp = before + '<zzzora:' + work + '>' + after
            v.prompt = temp.replace('<zzzora:', '<lora:')
        self.log('Fixed ' + str(replacements) + ' lora reference(s) containing broken paths...')


    # handles translating non-Auto1111 recognized samplers to Auto1111 samplers
    def verify_sampler(self, sampler):
        sampler = sampler.lower().strip()
        for a in self.auto1111_samplers:
            if sampler == a.lower().strip():
                return a

        # didn't find exact match, try known subs:
        if sampler == 'dpmpp_2m_sde_gpu':
            return 'DPM++ 2M SDE'
        elif sampler == 'dpmpp_2m_karras':
            return 'DPM++ 2M'
        elif sampler == 'dpmpp_3m_sde':
            return 'DPM++ 3M SDE'
        elif sampler == 'ddim_ddim_uniform':
            return 'DDIM'
        elif sampler == 'dpm++ 2m sde sgmuniform':
            return 'DPM++ 2M SDE'
        elif sampler == 'dpmpp_sde_karras':
            return 'DPM++ SDE'
        elif sampler == 'dpmpp_2s_ancestral_karras':
            return 'DPM++ 2S a'
        elif sampler == 'dpm++ 2m sde gpu':
            return 'DPM++ 2M SDE'
        elif sampler == 'dpmpp_3m_sde_gpu_karras':
            return 'DPM++ 3M SDE'
        elif sampler == 'dpmpp_2m_alt_karras':
            return 'DPM++ 2M'
        elif sampler == 'dpmpp_3m_sde_gpu':
            return 'DPM++ 3M SDE'
        elif sampler == 'euler_max':
            return 'Euler'
        elif sampler == 'dpmpp_2m_turbo':
            return 'DPM++ 2M'
        elif sampler == 'dpm++ 2m sde ays':
            return 'DPM++ 2M SDE'
        elif sampler == 'euler a turbo':
            return 'Euler a'
        elif sampler == 'dpmpp_sde_sgm_uniform':
            return 'DPM++ SDE'
        elif sampler == 'dpm++ 2m sgmuniform':
            return 'DPM++ 2M'
        elif sampler == 'dpmpp_3m_sde_karras':
            return 'DPM++ 3M SDE'
        elif sampler == 'dpmpp_2m_sde_karras':
            return 'DPM++ 2M SDE'
        elif sampler == 'ddim_sgm_uniform':
            return 'DDIM'
        elif sampler == 'dpm++ 2m turbo':
            return 'DPM++ 2M'
        elif sampler == 'dpmpp_sde':
            return 'DPM++ SDE'
        elif sampler == 'dpmpp_sde_gpu_karras':
            return 'DPM++ SDE'
        elif sampler == 'dpm_2_turbo':
            return 'DPM2'
        elif sampler == 'ddpm':
            return 'DPM2'
        elif sampler == 'euler_ancestral':
            return 'Euler a'
        elif sampler == 'dpmpp_3m_sde_gpu_sgm_uniform':
            return 'DPM++ 3M SDE'
        #elif sampler == 'sa solver':
        #    return ''
        else:
            if sampler != '':
                self.log('Warning: Couldn\'t find suitable sampler translation for ' + sampler + '; using default (DPM++ 2M)!', False)
            return 'DPM++ 2M'


    # handles logging to file/console
    def log(self, line, console = True):
        output = '[Prompts] > ' + str(line)
        if console:
            print(output)
        with open(self.logfile, 'a', encoding="utf-8") as f:
            f.write(output + '\n')
