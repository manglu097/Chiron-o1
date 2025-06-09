import math
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModel,
    AutoConfig
)

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def init_model(args):
    """
    Initializes and loads models specified in command-line arguments.

    Args:
        args: Parsed command-line arguments containing model paths.

    Returns:
        A dictionary where keys are model names (e.g., 'qwen25_vl_7b')
        and values are dictionaries containing the loaded 'model' and 'processor'.
        Returns an empty dictionary if no models are specified or loaded.
    """
    model_set = {}
    
    # --- Load Models Based on Provided Paths ---

    # Qwen/Qwen2.5-VL-72B-Instruct (Teacher)
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    # Qwen/Qwen2.5-VL-7B-Instruct (Intern)
    if args.qwen25_vl_7b_model_path:
        model_name = 'qwen25_vl_7b'
        print(f'Initializing {model_name}...')
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.qwen25_vl_7b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(args.qwen25_vl_7b_model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
            model_set[model_name] = {'model': model, 'processor': processor}
            print(f'{model_name} loaded successfully.')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")


    # Qwen/Qwen2-VL-7B-Instruct (Intern)
    if args.qwen2_vl_7b_model_path:
        model_name = 'qwen2_vl_7b'
        print(f'Initializing {model_name}...')
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.qwen2_vl_7b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(args.qwen2_vl_7b_model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
            model_set[model_name] = {'model': model, 'processor': processor}
            print(f'{model_name} loaded successfully.')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")



    # OpenGVLab/InternVL3-8B (Intern) - NEW
    if args.internvl3_8b_model_path:
        model_name = 'internvl3_8b'
        print(f'Initializing {model_name}...')
        try:
            device_map2 = split_model(args.internvl3_8b_model_path)
            model = AutoModel.from_pretrained(
                args.internvl3_8b_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                use_flash_attn=True, trust_remote_code=True, device_map=device_map2).eval()
            processor = AutoTokenizer.from_pretrained(args.internvl3_8b_model_path, trust_remote_code=True)
            processor.pad_token_id = processor.eos_token_id
        
            model_set[model_name] = {'model': model, 'processor': processor}
            print(f'{model_name} loaded successfully.')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    if not model_set:
        print("Warning: No models were loaded. Check provided model paths in arguments.")

    return model_set 