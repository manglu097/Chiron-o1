import json
from tqdm import tqdm
import os
import argparse
from utils import read_jsonl, get_chunk 
from model import init_model
from mics import MentorInternSearch 
import traceback

def mics_start(args):
    data_path = args.data_path
    data = None 
    if data_path.endswith('.jsonl'):
        try:
            data = read_jsonl(data_path)
        except Exception as e:
            print(f"Error loading jsonl data from {data_path}: {e}")
            return 
    elif data_path.endswith('.json'):
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading json data from {data_path}: {e}")
            return 
    else:
        print(f"Error: Unsupported data file format for {data_path}. Please use .jsonl or .json")
        return

    if data is None:
        print("Error: Data loading failed.")
        return

    output_path = args.output_path
    failed_search_path = args.output_path.replace('.jsonl', '_failed.jsonl')

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: 
             os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return 

    try:
        with open(output_path, "w") as search_file, open(failed_search_path, "w") as failed_search_file:
            if args.num_chunks > 1:
                try:
                    data = get_chunk(data, args.num_chunks, args.chunk_idx)
                except IndexError:
                    print(f"Error: Chunk index {args.chunk_idx} is out of range for {args.num_chunks} chunks.")
                    failed_search_file.write(json.dumps({"error": f"Invalid chunk index {args.chunk_idx} for {args.num_chunks} chunks."}) + "\n")
                    return

            # Initialize models
            model_set = None
            try:
                model_set = init_model(args)
            except Exception as e:
                print(f"Error initializing models: {e}")
                failed_search_file.write(json.dumps({"error": "Model initialization failed", "details": str(e)}) + "\n")
                return 

            # Instantiate the search class
            search_process = None
            search_process = MentorInternSearch(args)
            

            print(f"Processing {len(data)} data items...")
            for d in tqdm(data):
                data_id = d.get('rid', 'N/A') 
                try:
                    search_process.search(d, model_set, search_file, failed_search_file)
                except Exception as e:
                    error_message = traceback.format_exc()
                    print(f"\n!!! Unexpected error during search for data ID {data_id}: {error_message} !!!")
                    error_log = {
                        "rid": data_id,
                        "error": "Unhandled exception during search",
                        "details": str(e),
                    }
                    failed_search_file.write(json.dumps(error_log) + "\n")
                    failed_search_file.flush() 

    except IOError as e:
        print(f"Error opening or writing to output files ({output_path}, {failed_search_path}): {e}")
    except Exception as e:
         print(f"An unexpected error occurred outside the main loop: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mentor-Intern Reasoning Search") 
    parser.add_argument("--data_path", type=str, default='./rp_stage3_data/d2d_qa.jsonl', required=True, help="Path to input data (.jsonl or .json)") 
    parser.add_argument("--image_dir_path", type=str, default='./rp_stage3_data/images', required=True, help="Path to the directory containing images") 
    parser.add_argument("--output_path", type=str, default='./test/tsis_results.jsonl', required=True, help="Path to save successful results (.jsonl)") 
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to split data into")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to process (0-based)")
    parser.add_argument("--qwen25_vl_7b_model_path", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', required=True, help="Path to Qwen25-VL 7B model")
    parser.add_argument("--qwen2_vl_7b_model_path", type=str, default='Qwen/Qwen2-VL-7B-Instruct', required=True, help="Path to Qwen2-VL 7B model")
    parser.add_argument("--internvl3_8b_model_path", type=str, default='OpenGVLab/InternVL3-8B', required=True, help="Path to InternVL3 8B model")
    parser.add_argument("--openai_api_key", type=str, default="sk-xxx" ,required=True) 
    parser.add_argument("--openai_base_url", type=str, default='' ,required=True)
    parser.add_argument("--qwen_api_key", type=str, default="sk-xxx" ,required=True) 
    parser.add_argument("--qwen_base_url", type=str, default='' ,required=True)
    parser.add_argument("--gemini_api_key", type=str, default="sk-xxx" ,required=True) 
    parser.add_argument("--gemini_base_url", type=str, default='' ,required=True)
    parser.add_argument("--ds_api_key", type=str, default="sk-xxx" ,required=True) 
    parser.add_argument("--ds_base_url", type=str, default='' ,required=True)
    parser.add_argument("--mentor_models", nargs='+', type=str, default=['chatgpt-4o-latest', 'google/gemini-2.5-pro-preview-03-25', 'qwen2.5-vl-72b-instruct'], required=True, help="List of mentor model names (must match keys in model_dict or be 'gpt-...' style)")
    parser.add_argument("--intern_models", nargs='+', type=str, default=['qwen25_vl_7b', 'qwen2_vl_7b', 'internvl3_8b'], required=True, help="List of intern model names (must match keys in model_dict)")
    parser.add_argument("--evaluator_model", type=str, default='deepseek-chat', help="Evaluator model name (must be GPT for default JUDGE_PROMPT)")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum number of reasoning steps to generate.")
    parser.add_argument("--temperature1", type=float, default=1.2, help="First temperature for intern model evaluation and mentor generation.")
    parser.add_argument("--temperature2", type=float, default=0.2, help="Second temperature for intern model evaluation.")
  
    args = parser.parse_args()
  
    errors = []
    model_name_to_arg_map = {
        'qwen25_vl_7b': 'qwen25_vl_7b_model_path',
        'internvl3_8b': 'internvl3_8b_model_path',
        'qwen2_vl_7b': 'qwen2_vl_7b_model_path'
    }

    # Validate Mentors
    if len(args.mentor_models) < 2:
        errors.append("At least two mentor models must be specified via --mentor_models.")

    # Validate Interns
    if len(args.intern_models) < 1:
        errors.append("At least one intern model must be specified via --intern_models.")
    else:
        for model_name in args.intern_models:
            arg_name = model_name_to_arg_map.get(model_name)
            if not arg_name:
                 errors.append(f"Intern model '{model_name}' is not recognized or mapped to a path argument.")
            elif getattr(args, arg_name, None) is None:
                 errors.append(f"Path for intern model '{model_name}' (expected argument --{arg_name}) was not provided.")

    # Report errors if any
    if errors:
        print("\nArgument Validation Errors:")
        for error in errors:
            print(f"- {error}")
        parser.print_help() 
        exit(1) 

    mics_start(args)

