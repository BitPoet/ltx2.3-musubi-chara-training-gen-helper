import os
import argparse
import json
import time
import sys
import re

CONFIG_FILE = "base_config.json"

from typing import Any, Dict

def get_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                if isinstance(config, dict):
                    return config
        except Exception as e:
            print(f"Error loading {CONFIG_FILE}: {e}")
            return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def setup(args):
    config = get_config()
    changed = False
    
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
            sys.exit(1)
        config["CHECKPOINT_PATH"] = args.checkpoint
        changed = True
        
    if args.gemma is not None:
        if not os.path.exists(args.gemma):
            print(f"Error: Gemma path does not exist: {args.gemma}")
            sys.exit(1)
        config["GEMMA_PATH"] = args.gemma
        changed = True
        
    if changed:
        save_config(config)
        print(f"Updated {CONFIG_FILE}.")
    else:
        print("No valid paths provided for setup.")

def new_training(args):
    if not args.dataset:
        print("Error: --dataset is required for --new")
        sys.exit(1)
    if not args.name:
        print("Error: --name is required for --new")
        sys.exit(1)
        
    dataset_file = f"dataset_{args.name}.toml"
    training_file = f"training_args_{args.name}.toml"
    bat_file = f"train_{args.name}.bat"
    
    for f in [dataset_file, training_file, bat_file]:
        if os.path.exists(f):
            print(f"Error: Output file already exists and will not be overwritten: {f}")
            sys.exit(1)
            
    config = get_config()
    
    # Load defaults
    defaults = {
        "BLOCKS_TO_SWAP": 4,
        "NETWORKDIM": 64,
        "MAXSTEPS": 4000,
        "SAVEEVERY": 250,
        "GPU_ID": False,
        "LR": "7e-5"
    }
    
    for k, v in config.items():
        defaults[k] = v
        
    if args.blocks_to_swap is not None: defaults["BLOCKS_TO_SWAP"] = args.blocks_to_swap
    if args.network_dim is not None: defaults["NETWORKDIM"] = args.network_dim
    if args.max_steps is not None: defaults["MAXSTEPS"] = args.max_steps
    if args.save_every is not None: defaults["SAVEEVERY"] = args.save_every
    if args.gpu is not None: defaults["GPU_ID"] = args.gpu
    if args.lr is not None: defaults["LR"] = args.lr
    
    # Ensure required paths from config are present
    if "CHECKPOINT_PATH" not in defaults or not defaults["CHECKPOINT_PATH"]:
        print("Error: CHECKPOINT_PATH is not set. Run --setup first.")
        sys.exit(1)
    if "GEMMA_PATH" not in defaults or not defaults["GEMMA_PATH"]:
        print("Error: GEMMA_PATH is not set. Run --setup first.")
        sys.exit(1)
        
    ts = str(int(time.time()))
    try:
        network_dim = int(defaults.get("NETWORKDIM", 64))
    except ValueError:
        network_dim = 64
        
    network_alpha = network_dim // 2
    
    replace_dict = {
        "TS": ts,
        "NETWORKALPHA": str(network_alpha),
        "LORA_NAME": args.name,
    }
    
    for k, v in defaults.items():
        replace_dict[k.upper()] = str(v)
        
    dataset_path = args.dataset.replace("\\", "/")
    replace_dict["DATASET_PATH"] = dataset_path
    
    dataset_config_path = os.path.abspath(dataset_file).replace("\\", "/")
    replace_dict["DATASET_CONFIG_PATH"] = dataset_config_path
    
    gpu_id = defaults.get("GPU_ID", False)
    if gpu_id is not False and str(gpu_id).strip().lower() != "false" and str(gpu_id).strip() != "":
        replace_dict["GPU_OPTS"] = f"--gpu_ids {gpu_id}"
    else:
        replace_dict["GPU_OPTS"] = ""
        
    def replace_in_template(template_name, output_path):
        if not os.path.exists(template_name):
            print(f"Error: Template file {template_name} missing.")
            sys.exit(1)
            
        with open(template_name, "r", encoding="utf-8") as f:
            content = f.read()
            
        def replacer(match):
            key = match.group(1)
            if key in replace_dict:
                return replace_dict[key]
            return match.group(0)
            
        content = re.sub(r"##([A-Z_]+)##", replacer, content)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    replace_in_template("dataset_template.toml", dataset_file)
    replace_in_template("training_args_template.toml", training_file)
    replace_in_template("trainTEMPLATE.bat", bat_file)
    
    print(f"Generated {dataset_file}")
    print(f"Generated {training_file}")
    print(f"Generated {bat_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Musubi LTX-2 Training Generator")
    parser.add_argument("--setup", action="store_true", help="One-time setup")
    parser.add_argument("--checkpoint", type=str, help="Path to ltx-2.3 checkpoint")
    parser.add_argument("--gemma", type=str, help="Path to gemma text encoder")
    parser.add_argument("--new", action="store_true", help="Generate training setup")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--name", type=str, help="LoRA name")
    parser.add_argument("--blocks_to_swap", type=int, help="Override blocks_to_swap")
    parser.add_argument("--network_dim", type=int, help="Override network_dim")
    parser.add_argument("--max_steps", type=int, help="Override max_steps")
    parser.add_argument("--save_every", type=int, help="Override save_every")
    parser.add_argument("--gpu", type=str, help="Override gpu id")
    parser.add_argument("--lr", type=str, help="Override learning rate")
    
    args = parser.parse_args()
    
    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit(0)
        
    if args.setup:
        setup(args)
    elif args.new:
        new_training(args)
    else:
        print("Please specify --setup or --new. Run with -h for help.")
