# Copyright 2024, Bill Kennedy (https://github.com/rbbrdckybk/civitai-companion)
# SPDX-License-Identifier: MIT

import sys
import os
import glob
import argparse
from os.path import exists
from pathlib import Path
from scripts.utils import TextFile

# Handles all configuration options
# options specified in config file are overridden by options also specified on command line
class Config:
    def __init__(self, ap):
        self.logfile = os.path.join('logs', 'log.txt')
        os.makedirs('logs', exist_ok = True)

        # user options are populated here
        self.general_config = {}
        self.network_config = {}
        self.image_config = {}
        self.prompt_config = {}

        self.parser = ap
        self.options = None
        # reads user options from the command-line
        self.read_argparse()

        # commits user options from supplied config file if present
        if self.options.config_file != '':
            # read options from config file if supplied
            self.init_config(self.options.config_file)

        # commits user options from command line if present
        # (will override config file options)
        self.init_command_line()


    # for debugging, displays all user options
    def debug_display_user_options(self):
        print('')
        self.log('Image Options:')
        for k, v in self.image_config.items():
            self.log('  ' + str(k) + ' : ' + str(v))
        self.log('Network Options:')
        for k, v in self.network_config.items():
            self.log('  ' + str(k) + ' : ' + str(v))
        self.log('Prompt Options:')
        for k, v in self.prompt_config.items():
            self.log('  ' + str(k) + ' : ' + str(v))
        self.log('General Options:')
        for k, v in self.general_config.items():
            self.log('  ' + str(k) + ' : ' + str(v))
        print('')

    # handles committing options from user-supplied command-line arguments
    # only add if they're not already present (from config file) or not a default value
    def init_command_line(self):
        # network params
        if ('api_key' not in self.network_config) or (self.options.civitai_api_key != ''):
            self.network_config.update({'api_key' : self.options.civitai_api_key})
        if ('request_delay' not in self.network_config) or (self.options.civitai_request_delay != 1):
            self.network_config.update({'request_delay' : self.options.civitai_request_delay})
        if ('max_file_size' not in self.network_config) or (self.options.civitai_max_file_size != 1000000000):
            self.network_config.update({'max_file_size' : self.options.civitai_max_file_size})

        # image params
        if ('path' not in self.image_config) or (self.options.image_path != ''):
            self.image_config.update({'path' : self.options.image_path})
        if ('path_ignore_subdirs' not in self.image_config) or (self.options.image_path_ignore_subdirs != False):
            self.image_config.update({'path_ignore_subdirs' : self.options.image_path_ignore_subdirs})

        # prompt params
        if ('append_filename' not in self.prompt_config) or (self.options.prompt_append_filename != ''):
            self.prompt_config.update({'append_filename' : self.options.prompt_append_filename})
        if ('min_steps' not in self.prompt_config) or (self.options.prompt_min_steps != 0):
            self.prompt_config.update({'min_steps' : self.options.prompt_min_steps})
        if ('max_steps' not in self.prompt_config) or (self.options.prompt_max_steps != 0):
            self.prompt_config.update({'max_steps' : self.options.prompt_max_steps})
        if ('min_scale' not in self.prompt_config) or (self.options.prompt_min_scale != 0):
            self.prompt_config.update({'min_scale' : self.options.prompt_min_scale})
        if ('max_scale' not in self.prompt_config) or (self.options.prompt_max_scale != 0):
            self.prompt_config.update({'max_scale' : self.options.prompt_max_scale})
        if ('fix_resolution' not in self.prompt_config) or (self.options.prompt_fix_resolution != True):
            self.prompt_config.update({'fix_resolution' : self.options.prompt_fix_resolution})

        if ('only_include_base' not in self.prompt_config) or (self.options.prompt_only_include_base != ''):
            self.prompt_config.update({'only_include_base' : self.options.prompt_only_include_base})
        if ('output_template' not in self.prompt_config) or (self.options.prompt_output_template != ''):
            self.prompt_config.update({'output_template' : self.options.prompt_output_template})
        if ('output_header' not in self.prompt_config) or (self.options.prompt_output_header != ''):
            self.prompt_config.update({'output_header' : self.options.prompt_output_header})
        if ('output_footer' not in self.prompt_config) or (self.options.prompt_output_footer != ''):
            self.prompt_config.update({'output_footer' : self.options.prompt_output_footer})
        if ('output_save_as' not in self.prompt_config) or (self.options.prompt_output_save_as != ''):
            self.prompt_config.update({'output_save_as' : self.options.prompt_output_save_as})
        if ('word_filter_list' not in self.prompt_config) or (self.options.prompt_word_filter_list != ''):
            self.prompt_config.update({'word_filter_list' : self.options.prompt_word_filter_list})
        if ('neg_word_filter_list' not in self.prompt_config) or (self.options.prompt_neg_word_filter_list != ''):
            self.prompt_config.update({'neg_word_filter_list' : self.options.prompt_neg_word_filter_list})
        if ('lora_filter_list' not in self.prompt_config) or (self.options.prompt_lora_filter_list != ''):
            self.prompt_config.update({'lora_filter_list' : self.options.prompt_lora_filter_list})

        # general params
        if ('existing_model_path' not in self.general_config) or (self.options.existing_model_path != ''):
            self.general_config.update({'existing_model_path' : self.options.existing_model_path})
        if ('existing_lora_path' not in self.general_config) or (self.options.existing_lora_path != ''):
            self.general_config.update({'existing_lora_path' : self.options.existing_lora_path})
        if ('existing_embedding_path' not in self.general_config) or (self.options.existing_embedding_path != ''):
            self.general_config.update({'existing_embedding_path' : self.options.existing_embedding_path})
        if ('download_model_path' not in self.general_config) or (self.options.download_model_path != ''):
            self.general_config.update({'download_model_path' : self.options.download_model_path})
        if ('download_lora_path' not in self.general_config) or (self.options.download_lora_path != ''):
            self.general_config.update({'download_lora_path' : self.options.download_lora_path})
        if ('download_embedding_path' not in self.general_config) or (self.options.download_embedding_path != ''):
            self.general_config.update({'download_embedding_path' : self.options.download_embedding_path})


    # handles reading options from user-supplied command-line arguments
    def read_argparse(self):
        self.parser.add_argument(
            '--config_file',
            type=str,
            default='',
            help='configuration file'
        )
        self.parser.add_argument(
            '--civitai_api_key',
            type=str,
            default='',
            help='your civitai.com API key'
        )
        self.parser.add_argument(
            '--civitai_request_delay',
            type=float,
            default='1',
            help='minimum time between network requests to civitai.com (in seconds)'
        )
        self.parser.add_argument(
            '--civitai_max_file_size',
            type=int,
            default='1000000000',
            help='maximum file size to download in bytes (e.g.: 1000000000 = 1GB); 0 = no limit'
        )
        self.parser.add_argument(
            '--image_path',
            type=str,
            default='',
            help='path to folder containing images to scan for metadata'
        )
        self.parser.add_argument(
            '--image_path_ignore_subdirs',
            action='store_true',
            default=False,
            help='path to folder containing images to scan for metadata'
        )
        self.parser.add_argument(
            '--prompt_append_filename',
            type=str,
            default='',
            help='append this to each prompt''s assigned output filename'
        )
        self.parser.add_argument(
            '--prompt_min_steps',
            type=int,
            default=0,
            help='minimum allowed step count (0 = no limit)'
        )
        self.parser.add_argument(
            '--prompt_max_steps',
            type=int,
            default=0,
            help='maximum allowed step count (0 = no limit)'
        )
        self.parser.add_argument(
            '--prompt_min_scale',
            type=float,
            default='0',
            help='minimum allowed config scale (0 = no limit)'
        )
        self.parser.add_argument(
            '--prompt_max_scale',
            type=float,
            default='0',
            help='maximum allowed config scale (0 = no limit)'
        )
        self.parser.add_argument(
            '--prompt_fix_resolution',
            action='store_false',
            default=True,
            help='automatically adjust the prompt\'s width/height to whichever officially-supported base resolution most closely matches the image aspect ratio'
        )
        self.parser.add_argument(
            '--prompt_only_include_base',
            type=str,
            default='',
            help='comma-separated list of base models to include in prompt output (blank = all)'
        )
        self.parser.add_argument(
            '--prompt_output_template',
            type=str,
            default='',
            help='path/filename of prompt output template file'
        )
        self.parser.add_argument(
            '--prompt_output_header',
            type=str,
            default='',
            help='path/filename of header file to attach to prompt output file'
        )
        self.parser.add_argument(
            '--prompt_output_footer',
            type=str,
            default='',
            help='path/filename of footer file to attach to prompt output file'
        )
        self.parser.add_argument(
            '--prompt_output_save_as',
            type=str,
            default='',
            help='path/filename to save output file as'
        )
        self.parser.add_argument(
            '--prompt_word_filter_list',
            type=str,
            default='',
            help='comma-separated list of words to remove from prompts if found'
        )
        self.parser.add_argument(
            '--prompt_neg_word_filter_list',
            type=str,
            default='',
            help='comma-separated list of words to remove from negative prompts if found'
        )
        self.parser.add_argument(
            '--prompt_lora_filter_list',
            type=str,
            default='',
            help='comma-separated list of loras to remove if found (lora filename without path or extension)'
        )
        self.parser.add_argument(
            '--existing_model_path',
            type=str,
            default='',
            help='path to your existing model/checkpoint files'
        )
        self.parser.add_argument(
            '--existing_lora_path',
            type=str,
            default='',
            help='path to your existing LoRA files'
        )
        self.parser.add_argument(
            '--existing_embedding_path',
            type=str,
            default='',
            help='path to your existing embedding files'
        )
        self.parser.add_argument(
            '--download_model_path',
            type=str,
            default='',
            help='path that downloaded model/checkpoint files will be saved to'
        )
        self.parser.add_argument(
            '--download_lora_path',
            type=str,
            default='',
            help='path that downloaded LoRA files will be saved to'
        )
        self.parser.add_argument(
            '--download_embedding_path',
            type=str,
            default='',
            help='path that downloaded embedding files will be saved to'
        )

        self.options = self.parser.parse_args()


    # handles committing options from user-specified config file
    def init_config(self, config_file):
        if not exists(config_file):
            self.log('Error: specified config file ' + config_file + ' does not exist; using defaults instead!')
            return None

        file = TextFile(config_file)
        if file.lines_remaining() > 0:
            self.log("Reading configuration from " + config_file + "...")
            while file.lines_remaining() > 0:
                line = file.next_line()

                # update config values for found directives
                if '=' in line:
                    line = line.split('=', 1)
                    command = line[0].lower().strip()
                    value = line[1].strip()

                    if command == 'civitai_api_key':
                        if value != '':
                            self.network_config.update({'api_key' : value})

                    elif command == 'civitai_request_delay':
                        if value != '':
                            try:
                                float(value)
                            except:
                                pass
                            else:
                                self.network_config.update({'request_delay' : float(value)})

                    elif command == 'civitai_max_file_size':
                        if value != '':
                            try:
                                int(value)
                            except:
                                pass
                            else:
                                self.network_config.update({'max_file_size' : int(value)})

                    elif command == 'image_path':
                        if value != '':
                            self.image_config.update({'path' : value})

                    elif command == 'image_path_ignore_subdirs':
                        if value == 'yes' or value == 'true':
                            self.image_config.update({'path_ignore_subdirs' : True})
                        elif value == 'no' or value == 'false':
                            self.image_config.update({'path_ignore_subdirs' : False})

                    elif command == 'prompt_append_filename':
                        if value != '':
                            self.prompt_config.update({'append_filename' : value})

                    elif command == 'prompt_min_steps':
                        if value != '':
                            try:
                                int(value)
                            except:
                                pass
                            else:
                                self.prompt_config.update({'min_steps' : int(value)})

                    elif command == 'prompt_max_steps':
                        if value != '':
                            try:
                                int(value)
                            except:
                                pass
                            else:
                                self.prompt_config.update({'max_steps' : int(value)})

                    elif command == 'prompt_min_scale':
                        if value != '':
                            try:
                                float(value)
                            except:
                                pass
                            else:
                                self.prompt_config.update({'min_scale' : float(value)})

                    elif command == 'prompt_max_scale':
                        if value != '':
                            try:
                                float(value)
                            except:
                                pass
                            else:
                                self.prompt_config.update({'max_scale' : float(value)})

                    elif command == 'prompt_fix_resolution':
                        if value == 'yes' or value == 'true':
                            self.prompt_config.update({'fix_resolution' : True})
                        elif value == 'no' or value == 'false':
                            self.prompt_config.update({'fix_resolution' : False})

                    elif command == 'prompt_only_include_base':
                        if value != '':
                            self.prompt_config.update({'only_include_base' : value})

                    elif command == 'prompt_output_template':
                        if value != '':
                            self.prompt_config.update({'output_template' : value})

                    elif command == 'prompt_output_header':
                        if value != '':
                            self.prompt_config.update({'output_header' : value})

                    elif command == 'prompt_output_footer':
                        if value != '':
                            self.prompt_config.update({'output_footer' : value})

                    elif command == 'prompt_output_save_as':
                        if value != '':
                            self.prompt_config.update({'output_save_as' : value})

                    elif command == 'prompt_word_filter_list':
                        if value != '':
                            self.prompt_config.update({'word_filter_list' : value})

                    elif command == 'prompt_neg_word_filter_list':
                        if value != '':
                            self.prompt_config.update({'neg_word_filter_list' : value})

                    elif command == 'prompt_lora_filter_list':
                        if value != '':
                            self.prompt_config.update({'lora_filter_list' : value})

                    elif command == 'existing_model_path':
                        if value != '':
                            self.general_config.update({'existing_model_path' : value})

                    elif command == 'existing_lora_path':
                        if value != '':
                            self.general_config.update({'existing_lora_path' : value})

                    elif command == 'existing_embedding_path':
                        if value != '':
                            self.general_config.update({'existing_embedding_path' : value})

                    elif command == 'download_model_path':
                        if value != '':
                            self.general_config.update({'download_model_path' : value})

                    elif command == 'download_lora_path':
                        if value != '':
                            self.general_config.update({'download_lora_path' : value})

                    elif command == 'download_embedding_path':
                        if value != '':
                            self.general_config.update({'download_embedding_path' : value})



    # handles logging to file/console
    def log(self, line, console = True):
        output = '[Config] > ' + str(line)
        if console:
            print(output)
        with open(self.logfile, 'a', encoding="utf-8") as f:
            f.write(output + '\n')
