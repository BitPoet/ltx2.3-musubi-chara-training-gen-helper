# Quick Musubi LTX-2.3 Training Generator

A helper to generate training configs for LTX-2.3 character LoRAs using the
[LTX-2 fork of musubi tuner](https://github.com/AkaneTendo25/musubi-tuner).

Creates config files and a .bat file to run the full training.

Currently works on Windows, pull requests for Linux support are welcome.

A working installation of musubi ltx-2 fork is needed.

## Usage

### Installation

Download or clone the files of the repository into a new folder "training" inside your musubi directory.

### One-Time Setup

Run ```training_gen.py --setup --checkpoint PATH-TO-LTX-2.3-DEV --gemma PATH-TO-GEMMA-TEXT-ENCODER```.

If you enter a non-existing path to a file, you will receive an error.

The values will be saved in base_settings.json. If you move your models, rerun ```training_gen.py --setup``` or edit base_settings.json directly.

### Generate Training Setup

Open a command shell and cd into musubi folder.

Activate the venv.

```cd``` into the *trainings* folder.

Run ```.\training_gen.py --new --dataset PATH-TO-DATASET --name LORA-NAME```

This will generate:

* A file dataset_LORA-NAME.toml in the current directory
* A file training_args_LORA-NAME.toml in the current directory
* A file train_LORA-NAME.bat in the current directory

#### Additional Parameters

You can override some default parameters when you set up training:

* blocks_to_swap N
  Defaults to 4, you can lower that to speed up training if you have enough VRAM.
* network_dim N
  Defaults to 64, lower value means smaller LoRA, you can experiment with setting
  this to 32 or 16.
* max_steps N
  Defaults to 4000, the LoRA often converges sufficiently around 3000 steps.
* save_every N
  Defaults to 250, but if you want to save space, you can raise this number.
  This saves a checkpoint of the LoRA every N training steps, so you can
  test how well the LoRA works at different steps and cherry-pick the one
  with best prompt adherence and visual identity.
* gpu N
  If you have multiple GPUs, you may need to bind musubi to a specific GPU.
  This is the same numeric id (0, 1, ...) you'd use for ComfyUI.
* lr FLOAT
  Learning rate. Defaults to 7e-5, which has been found to work well.


### Train

From within the *trainings* folder, run ```.\trainLORA-NAME.bat```

This will run all three steps of training with musubi in sequence:

* Cache image latents
* Cache text embeddings
* Run the actual training

#### Output Files

_LoRA_:

The LoRA (and its intermediate checkpoints) will be written to the subdirectory
"output" underneath the dataset directory.

_Logs_:

The training logs in tensorflow format will be written to the subdirectory
"logs" underneath the dataset directory.

### Progress

If you have tensorboard installed with musubi, you can watch the training graphs
by running "showlogs.bat LORA-NAME".

## Technical Details

### Placeholders in template files

* CHECKPOINT_PATH
  The full path to the ltx-2.3-22b-dev.safetensors file.
* GEMMA_PATH
  The full path to the directory in which gemma-3-12b-it-qat-q4_0-unquantized is stored.
* DATASET_CONFIG_PATH
  Full path to the dataset_LORA_NAME.toml file, automatically set.
* GPU_ID
  Integer ID of the GPU to use. Will be added as "--gpu_ids GPU_ID" to trainLORA-NAME.bat.
* DATASET_PATH
  Directory where the dataset is, both images and text files.
* TS
  A timestamp to make sure that the latent cache is recreated on consecutive runs.
* MAXSTEPS
  Number of steps to train.
* SAVEEVERY
  The trainer will save a copy of the current LoRA at this interval of steps so you
  can cherry-pick the best result in case the full LoRA got overbaked.
* NETWORKDIM
  Network dimension, which influences LoRA size (higher = bigger) and defines a maximum
  window of parameters in the base model that can be affected by the LoRA. Defaults to
  64 and must be a power of 2. 32 or 16 may also work.
* NETWORKALPHA
  Alpha dimension, which limits the "weight" of the training so the LoRA isn't as easily
  overfit. Automatically set to half of NETWORKDIM, which is generally considered a safe
  value.
  
