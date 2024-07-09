# Copyright 2024, Bill Kennedy (https://github.com/rbbrdckybk/civitai-companion)
# SPDX-License-Identifier: MIT

import sys
import os
import re
from collections import deque
from os.path import exists
from pathlib import Path

# for easy reading of prompt/config files
class TextFile():
    def __init__(self, filename):
        self.lines = deque()
        if exists(filename):
            with open(filename, encoding = 'utf-8') as f:
                l = f.readlines()

            for x in l:
                # remove newline and whitespace
                x = x.strip('\n').strip();
                # remove comments
                x = x.split('#', 1)[0].strip();
                if x != "":
                    # these lines are actual prompts
                    self.lines.append(x)

    def next_line(self):
        return self.lines.popleft()

    def lines_remaining(self):
        return len(self.lines)


# given a filename, returns a filesystem-safe version with illegal chars replaced
def sanitize_filename(filename):
    safe_filename = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", filename)
    return safe_filename


# fixes common formatting issues in user prompts
def sanitize_prompt(p):
    # remove newlines
    split_text = p.splitlines()
    p = ''.join(split_text)
    # remove explicit embedding declarations
    p = p.replace('embedding:', '')
    # remove common readability/formatting issues
    while '  ' in p:
        p = p.replace('  ', ' ')
    while ',,' in p:
        p = p.replace(',,', ',')
    while ' ,' in p:
        p = p.replace(' ,', ',')
    while '.,' in p:
        p = p.replace('.,', ',')
    while '. ,' in p:
        p = p.replace('. ,', ',')
    while ',.' in p:
        p = p.replace(',.', ',')
    while ', .' in p:
        p = p.replace(', .', ',')
    while ', ,' in p:
        p = p.replace(', ,', ',')
    while '8 k' in p:
        p = p.replace('8 k', '8k')
    while '4 k' in p:
        p = p.replace('4 k', '4k')
    # regex to force a space after commas and periods (except in decimal #s)
    #p = re.sub(r'(?<=[.,])(?=[^\s])', r' ', p).strip()
    p = re.sub(r'(?<=[,])(?=[^\s])', r' ', p).strip()
    p = re.sub('\.(?!\s|\d|$)', '. ', p).strip()
    while ', ,' in p:
        p = p.replace(', ,', ',')
    # remove leading spaces and commas
    while p.startswith(' ') or p.startswith(','):
        p = p[1:]
    # remove trailing spaces and commas
    while p.endswith(' ') or p.endswith(','):
        p = p[:-1]
    return p


# case-insensitive word replace
# will only replace if a case-insensitive match on the word is found AND
# the word is preceeded by a space, comma, or the start of the string
# AND the word is followed by a space, comma, period, or the end of the string
def word_replace(word, text):
    if word.lower().strip() == text.lower().strip():
        return ''

    final = text
    # this should be all that's required; not sure why the ^ $ aren't matching start/end of string...
    final = re.sub("(?<=[, ^])" + re.escape(word) + "(?=[\., $])", "", text, flags=re.IGNORECASE)

    if final.lower().strip().startswith(word.lower().strip()):
        final = final[len(word):]

    if final.lower().strip().endswith(word.lower().strip()):
        final = final[:0-len(word)]

    return final


# case-insensitive replace
def ireplace(old, new, text):
    idx = 0
    while idx < len(text):
        index_l = text.lower().find(old.lower(), idx)
        if index_l == -1:
            return text
        text = text[:index_l] + new + text[index_l + len(old):]
        idx = index_l + len(new)
    return text


# for EXIF metadata formatted by Dream Factory,
# extracts model filename from full identifier string
def extract_model_filename(model_id):
    filename = model_id
    if '[' in filename:
        filename = filename.split('[', 1)[0].strip()
    filename = os.path.basename(filename)
    if filename.endswith('.safetensors'):
        filename = filename[:-12]
    return filename


# for EXIF metadata formatted by Dream Factory,
# extracts model hash from full identifier string if present
def extract_model_hash(model_id):
    hash = ''
    if '[' in model_id and ']' in model_id:
        hash = model_id.split('[', 1)[1].strip()
        hash = hash.split(']', 1)[0].strip()
    return hash


# for EXIF metadata formatted by Dream Factory,
# extracts SD parameters from the full command
def extract_params_from_command(command):
    params = {
        'prompt' : "",
        'seed' : "",
        'width' : "",
        'height' : "",
        'steps' : "",
        'scale' : "",
        'input_image' : "",
        'strength' : "",
        'neg_prompt' : "",
        'model' : "",
        'sampler' : "",
        'styles' : "",
        'clip_skip' : ""
    }

    if command != "":
        command = command.strip('"')

        # need this because of old format w/ upscale info included
        if '(upscaled' in command:
            command = command.split('(upscaled', 1)[0]
            command = command.replace('(upscaled', '')

        if '--prompt' in command:
            temp = command.split('--prompt', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'prompt' : temp.strip().strip('"')})

        else:
            # we'll assume anything before --ddim_steps is the prompt
            temp = command.split('--ddim_steps', 1)[0]
            temp = temp.strip()
            if temp is not None and len(temp) > 0 and temp[-1] == '\"':
                temp = temp[:-1]
            temp = temp.replace('\\', '')
            params.update({'prompt' : temp})

        if '--neg_prompt' in command:
            temp = command.split('--neg_prompt', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'neg_prompt' : temp.strip().strip('"')})

        if '--ckpt' in command:
            temp = command.split('--ckpt', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'model' : temp.strip().strip('"')})

        if '--sampler' in command:
            temp = command.split('--sampler', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'sampler' : temp.strip()})

        if '--ddim_steps' in command:
            temp = command.split('--ddim_steps', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'steps' : temp.strip()})

        if '--scale' in command:
            temp = command.split('--scale', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'scale' : temp.strip()})

        if '--seed' in command:
            temp = command.split('--seed', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'seed' : temp.strip()})

        if '--W' in command:
            temp = command.split('--W', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'width' : temp.strip()})

        if '--H' in command:
            temp = command.split('--H', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'height' : temp.strip()})

        if '--init-img' in command:
            temp = command.split('--init-img', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            temp = temp.replace('../', '').strip().strip('"')
            head, tail = os.path.split(temp)
            if tail == '':
                tail = temp
            params.update({'input_image' : tail})

        if '--strength' in command:
            temp = command.split('--strength', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'strength' : temp.strip()})

        if '--clip-skip' in command:
            temp = command.split('--clip-skip', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'clip_skip' : temp.strip()})

        if '--styles' in command:
            temp = command.split('--styles', 1)[1]
            if '--' in temp:
                temp = temp.split('--', 1)[0]
            params.update({'styles' : temp.strip()})

    return params
