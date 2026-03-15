@echo off
setlocal disabledelayedexpansion

set TRAININGDIR=%~dp0
cd /d "%TRAININGDIR%"

cd ..
set MUSUBIDIR="%cd%"

echo on
@echo ===============
@echo Activating venv
@echo ===============
call .\venv\Scripts\activate.bat

@echo ===============
@echo Caching Latents
@echo ===============
accelerate launch ##GPU_OPTS## "%MUSUBIDIR%\ltx2_cache_latents.py" --dataset_config "%TRAININGDIR%\dataset_##LORA_NAME##.toml" --ltx2_checkpoint "##CHECKPOINT_PATH##" 

@echo ==================
@echo Caching Embeddings
@echo ==================
accelerate launch ##GPU_OPTS## "%MUSUBIDIR%\ltx2_cache_text_encoder_outputs.py" --dataset_config "%TRAININGDIR%\dataset_##LORA_NAME##.toml" --ltx2_checkpoint "##CHECKPOINT_PATH##" --gemma_root "##GEMMA_PATH##" --gemma_load_in_8bit --skip_existing

@echo =================
@echo Starting Training
@echo =================
accelerate launch ##GPU_OPTS## "%MUSUBIDIR%\ltx2_train_network.py" --config_file "%TRAININGDIR%\training_args_##LORA_NAME##.toml" --ltx2_checkpoint "##CHECKPOINT_PATH##" 
