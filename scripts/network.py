# Copyright 2024, Bill Kennedy (https://github.com/rbbrdckybk/civitai-companion)
# SPDX-License-Identifier: MIT

import sys
import os
import requests
import shutil
import time
from tqdm.auto import tqdm
from os.path import exists
from pathlib import Path
from scripts.utils import TextFile

class Network:
    # config is a dict of options prepared by the Config class
    def __init__(self, config):
        # backup existing log file & create new one
        self.logfile = os.path.join('logs', 'log.txt')
        os.makedirs('logs', exist_ok = True)

        self.api_key = config.get('api_key')
        self.request_delay = config.get('request_delay')
        # file size download limit in bytes (2500000000 = 2.5GB)
        self.max_file_size = config.get('max_file_size')

        self.last_request_time = time.perf_counter() - self.request_delay

        # list of civitai.com version IDs of resources that should not be downloaded
        self.do_not_download = self.init_do_not_download()

    # waits the minimum specified time between requests, if necessary
    def network_pause(self):
        #self.log('Waiting until ' + str(self.request_delay) + ' secs have passed since last request...')
        while time.perf_counter() - self.last_request_time < self.request_delay:
            time.sleep(0.1)


    # sets up list of resources that the user does not want to download
    def init_do_not_download(self):
        ids = []
        if self.file_exists('cache', 'do_not_download.txt'):
            pf = os.path.join('cache', 'do_not_download.txt')
            file = TextFile(pf)
            if file.lines_remaining() > 0:
                self.log('Caching "do not download" list from ' + str(pf) + '...', False)
                while file.lines_remaining() > 0:
                    line = file.next_line().strip()
                    try:
                        int(line)
                    except:
                        pass
                    else:
                        ids.append(line)
        self.log('Cached ' + str(len(ids)) + ' resource IDs that will not be downloaded.', False)
        return ids


    # downloads a file from the given url
    # local_filename can be optionally specified, otherwise will attempt to discern it
    def download_file(self, url, local_filepath='', local_filename=''):
        vid = url.rsplit('/', 1)[1]
        # check do not download list
        if vid in self.do_not_download:
            self.log('This ID ('+ str(vid) + ') is in the \'do not download\' list; aborting download!')
            return
        # check if the designated ouput file already exists
        if self.file_exists(local_filepath, local_filename):
            self.log('Error: ' + os.path.join(local_filepath, local_filename) + ' already exists; aborting download!')
        else:
            self.network_pause()
            # add API key to the request header if present
            headers = {}
            if self.api_key != '':
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                }
            # make the request
            self.last_request_time = time.perf_counter()
            with requests.get(url, stream=True, headers=headers) as r:
                if r.status_code == 200:
                    # attempt to get the filename from the response header
                    if 'Content-Disposition' in r.headers:
                        filename = r.headers.get('Content-Disposition')
                        if 'filename=' in filename:
                            remote_filename = filename.split('filename=', 1)[1].strip('\"')
                            if local_filename == '':
                                local_filename = remote_filename
                            else:
                                if local_filename != remote_filename:
                                    self.log('Warning: remote filename (' + remote_filename  + ') doesn\'t match expected filename (' + local_filename + ')!')
                    if local_filename != '':
                        full_output_path = os.path.join(local_filepath, local_filename)
                        # check if the designated ouput file already exists
                        if self.file_exists(local_filepath, local_filename):
                            self.log('Error: ' + full_output_path  + ' already exists; aborting download!')
                        else:
                            if local_filepath != '':
                                # create output dir if necessary
                                os.makedirs(local_filepath, exist_ok = True)
                            file_size = int(r.headers.get('Content-Length', 0))
                            if (self.max_file_size == 0) or (file_size <= self.max_file_size):
                                # start the download
                                self.log('Downloading from ' + url + '...')
                                desc = "(Unknown total file size)" if file_size == 0 else ""
                                with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
                                    with open(full_output_path, 'wb') as f:
                                        shutil.copyfileobj(r_raw, f)
                            else:
                                self.log('Warning: ' + local_filename + ' (' + str(file_size) + ' bytes) is over the maximum file size limit of ' + str(self.max_file_size) + ' bytes; download aborted!')
                    else:
                        self.log('Error: unable to determine output filename when downloading from ' + url + '; aborting download!')
                else:
                    local_filename = ''
                    if 'reason=download-auth' in r.url or r.status_code == 401:
                        self.log('Error: an API key is required to download from ' + url + ' (response code: ' + str(r.status_code) + ')!')
                    elif r.status_code == 403:
                        self.log('Error: the model at this URL is currently in early-access and unavailable: ' + url + ' (response code: ' + str(r.status_code) + ')!')
                    else:
                        self.log('Error: unable to download from ' + url + ' (response code: ' + str(r.status_code) + ')!')
        return local_filename

    # returns true if the given path/filename combo already exists, false otherwise
    def file_exists(self, local_filepath, local_filename):
        if local_filename == '':
            return False
        else:
            full_path = os.path.join(local_filepath, local_filename)
            return os.path.isfile(full_path)

    # handles logging to file/console
    def log(self, line, console = True):
        output = '[Network] > ' + str(line)
        if console:
            print(output)
        with open(self.logfile, 'a', encoding="utf-8") as f:
            f.write(output + '\n')
