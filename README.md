# Civitai Companion

 Utility for extracting prompt metadata from Civitai AI images and auto-downloading resrouces used to generate them.

# Features

 * Batch extracts prompts and all associated metadata (dimensions, model, sampler, steps, scale, etc) from images downloaded from civitai.com.
 * Automatically batch downloads necessary resources (models, LoRAs, embeddings) required to generate downloaded images. You may select the type(s) of resources that you want downloaded, a maximum file size for downloads, and blacklist resources that you never want downloaded.
 * Outputs all prompt information into user-specified format driven by simple template system (includes sample template for [Dream Factory](https://github.com/rbbrdckybk/dream-factory)).
 * LoRA references embedded in prompts that have user-specified paths are automatically modified to match your path structure.
 * LoRAs that are referenced in the 'resources' section of Civitai's prompt metadata but aren't embedded in the actual prompt are automatically added.
 * Optional specification of min/max values for certain prompt metadata values (e.g.: you may set max steps to 60, values over this will be set to 60).
 * Optional filtering for automatic removal of unwanted words from positive/negative prompts.
 * Optional filtering for automatic removal of unwanted LoRAs from prompts.
 * Optional image dimension re-sizing to the closest "official" resolution for the base model that each image is generated from (while preserving aspect ratio).
 * Samplers that are unsupported in Automatic1111's webui are automatically translated into an appropriate supported sampler.

# Requirements

You'll just need a working Python 3.10 environment. The setup instructions below assume that you're working within a suitable environment.

# Setup

**[1]** Navigate to wherever you want Civitai Companion to live, then clone this repository and switch to its directory:
```
git clone https://github.com/rbbrdckybk/civitai-companion
cd civitai-companion
```

If you're already using [Dream Factory](https://github.com/rbbrdckybk/dream-factory), you should be good to go and can probably skip this step.

**[2]** Install a few required packages:
```
pip install --no-input requests tqdm pillow
```

# Usage

You can verify that everything works properly by running (or, you can simply type **go** if you're running Windows):
```
python civitai_reader.py --config_file config-example.txt
```

If everything is working, you should see Civitai Companion start, scan the 7 sample images in the **img-examples** directory, extract their metadata, and then automatically download several LoRAs that are referenced by them.

You'll see errors that a few couldn't be downloaded because they require a civitai.com API key. You can [read Civitai's guide here](https://education.civitai.com/civitais-guide-to-downloading-via-api/) to get your own API key (skip to the "How do I get an API token/key?" section). After you have an API key, add it to your **config-example.txt** file and re-run Civitai Companion. You should see that the LoRAs that previously could not be downloaded complete successfully now.

Read through the rest of the **config-example.txt** file and modify settings to suit your needs.