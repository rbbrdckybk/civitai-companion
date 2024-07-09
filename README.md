# Civitai Companion

 Utility for extracting prompt metadata from Civitai AI images, auto-downloading the resources used to generate them, and outputting/formatting the prompt information in a manner defined by user-created templates.
 
 This was originally conceived as a companion tool for [Dream Factory](https://github.com/rbbrdckybk/dream-factory), so the included example template will work with Dream Factory out of the box. Templates can easily be created for use with Automatic1111 and other tools.

# Features

 * Batch extracts prompts and all associated metadata (dimensions, model, sampler, steps, scale, etc) from images downloaded from civitai.com.
 * Automatically batch downloads necessary resources (models, LoRAs, embeddings) required to generate downloaded images. You may select the type(s) of resources that you want downloaded, a maximum file size for downloads, and blacklist resources that you never want downloaded.
 * Outputs all extracted prompt information and associated metadata into user-specified format driven by simple template system (includes sample template for [Dream Factory](https://github.com/rbbrdckybk/dream-factory)).
 * LoRA references embedded in prompts that have user-specified paths are automatically modified to match your path structure (e.g.: ```<lora:some\user\specified\path\lora_name:1.0>``` becomes ```<lora:lora_name:1.0>```).
 * LoRAs that are referenced in the 'resources' section of Civitai's prompt metadata but aren't embedded in the actual prompt are automatically added.
 * Optional specification of min/max values for certain prompt metadata values (e.g.: you may set max steps to 60, values over this will be set to 60).
 * Optional filtering for automatic removal of unwanted words from positive/negative prompts.
 * Optional filtering for automatic removal of unwanted LoRAs from prompts.
 * Optional image dimension re-sizing to the closest "official" resolution for the base model that each image is generated from (while preserving aspect ratio).
 * Sampler names that are unsupported in Automatic1111's webui are automatically translated into an appropriate supported sampler (e.g.: 'dpmpp_3m_sde' becomes 'DPM++ 3M SDE').

# Requirements

You'll just need a working Python environment (tested on 3.10). The setup instructions below assume that you're working within a suitable environment.

# Setup

**[1]** Navigate to wherever you want Civitai Companion to live, then clone this repository and switch to its directory:
```
git clone https://github.com/rbbrdckybk/civitai-companion
cd civitai-companion
```

If you're already using [Dream Factory](https://github.com/rbbrdckybk/dream-factory), you should be good to go and can probably skip the following step.

**[2]** Install a few required packages:
```
pip install --no-input requests tqdm pillow
```

# Usage

You can verify that everything works properly by running (or, you can simply type **start** if you're running Windows):
```
python civitai_reader.py --config_file config-example.txt
```

If everything is working, you should see Civitai Companion start, scan the seven sample images in the **img-examples** directory, extract their metadata, automatically download several LoRAs that are referenced by them, and create a **civitai_[date]_[time].prompts** output file that is usable by [Dream Factory](https://github.com/rbbrdckybk/dream-factory) to create new images similar to the seven samples (check the **template-example.txt** file in the **inc** folder if you want to make changes to the output format for a different tool).

You'll notice errors that a few LoRAs couldn't be downloaded because they require a civitai.com API key. You can [read Civitai's guide here](https://education.civitai.com/civitais-guide-to-downloading-via-api/) to get your own API key (skip to the "How do I get an API token/key?" section). After you have an API key, add it to your **config-example.txt** file and re-run Civitai Companion. You should see that the LoRAs that previously could not be downloaded complete successfully now.

Read through the rest of the **config-example.txt** file and modify settings to suit your needs.

# Template Reference

The following tokens may be placed into template files; Civitai Companion will replace these tokens with the values extracted from image metadata.

An example template file (**template-example.txt**) is located in the **inc** folder and will output prompt files compatible with [Dream Factory](https://github.com/rbbrdckybk/dream-factory).

 * ```[PROMPT]``` The positive prompt used to produce this image.
 * ```[PROMPT_RAW]``` The positive prompt before any filters are applied.
 * ```[NEG_PROMPT]``` The negative prompt used to produce this image.
 * ```[NEG_PROMPT_RAW]``` The negative prompt before any filters are applied.
 * ```[MODEL]``` The model used to produce this image.
 * ```[SEED]``` The seed value used to produce this image.
 * ```[SAMPLER]``` The sampler used to produce this image.
 * ```[CLIP_SKIP]``` The clip skip setting used to produce this image.
 * ```[WIDTH]``` The width of this image in pixels.
 * ```[HEIGHT]``` The height of this image in pixels.
 * ```[SCALE]``` The cfg scale value used to produce this image.
 * ```[STEPS]``` The number of steps used to produce this image.
 * ```[BASE_MODEL]``` The base model (e.g.: SDXL 1.0, SD 1.5, Pony, etc) that the [MODEL] is based on.
 * ```[MODEL_HASH]``` The hash of the [MODEL].
 * ```[FILENAME]``` The original filename that this metadata was extracted from.
 * ```[FILEPATH]``` The original full path and filename that this metadata was extracted from.
 * ```[REF_NUM]``` A sequential reference number assigned to this image at processing time.
 * ```[RAW_METADATA]``` The full raw (unprocessed) metadata extracted from this image.

# Advanced Usage Tips

 * Configuration values may be specified in a config file and passed to Civitai Companion via the ```--config_file``` argument as demonstrated in the usage example **or** values may be passed via the command line. Values passed via command-line arguments override values set in a config file. This may be useful if you want to share a single config file between multiple workflows and simply override a few values on the command line for each.
 * If you need to do any troubleshooting or encounter errors, there is a **log.txt** file in the **logs** folder that contains details of the last run in much greater verbosity than what is displayed on the console. 
 * If you need to blacklist (e.g.: prevent from ever being downloaded again) a resource for whatever reason, add its civitai.com version ID to the **do_not_download.txt** file in the **cache** folder. If the resource has already been downloaded, its version ID should be in the **civitai_version_ids.txt** file. Simply search for the resource's name and copy the number at the start of the line (before the first comma) into the do_not_download.txt file.
 