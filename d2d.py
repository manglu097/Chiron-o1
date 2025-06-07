import json
from tqdm import tqdm
import os
import argparse
from utils import read_jsonl, get_chunk 
from model import init_model
from comcts import TeacherStudentSearch 


def start_tsis(args):
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
            search_process = TeacherStudentSearch(args)
            

            print(f"Processing {len(data)} data items...")
            for d in tqdm(data):
                data_id = d.get('rid', 'N/A') 
                try:
                    search_process.search(d, model_set, search_file, failed_search_file)
                except Exception as e:
                    print(f"\n!!! Unexpected error during search for data ID {data_id}: {e} !!!")
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
    # parser = argparse.ArgumentParser(description="Run Teacher-Student Reasoning Search") 
    # parser.add_argument("--data_path", type=str, default='./rp_stage3_data/tsis_data.jsonl', help="Path to input data (.jsonl or .json)") 
    # parser.add_argument("--image_dir_path", type=str, default='./rp_stage3_data/images', help="Path to the directory containing images") 
    # parser.add_argument("--output_path", type=str, default='./output/tsis_results.jsonl', help="Path to save successful results (.jsonl)") 
    # parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to split data into")
    # parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to process (0-based)")

    # # --- Model Path Arguments ---
    # parser.add_argument("--qwen25_vl_72b_model_path", type=str, default='/mnt/petrelfs/sunhaoran1/Radio--MCTS/model/Qwen2.5-VL-72B-Instruct', help="Path to Qwen25-VL 72B model")
    # parser.add_argument("--qwen25_vl_7b_model_path", type=str, default='/mnt/petrelfs/sunhaoran1/Radio--MCTS/model/Qwen2.5-VL-7B-Instruct', help="Path to Qwen25-VL 7B model")
    # parser.add_argument("--qwen2_vl_7b_model_path", type=str, default='/mnt/petrelfs/sunhaoran1/Radio--MCTS/model/Qwen2-VL-7B-Instruct', help="Path to Qwen2-VL 7B model")
    # parser.add_argument("--internvl3_78b_model_path", type=str, default='/mnt/petrelfs/sunhaoran1/Radio--MCTS/model/InternVL3-78B', help="Path to InternVL3 78B model")
    # parser.add_argument("--internvl3_8b_model_path", type=str, default='OpenGVLab/InternVL3-8B', help="Path to InternVL3 8B model")
    # parser.add_argument("--internvl3_9b_model_path", type=str, default='OpenGVLab/InternVL3-9B', help="Path to InternVL3 9B model")
    # parser.add_argument("--minicpm_o_2_6_model_path", type=str, default='openbmb/MiniCPM-o-2_6', help="Path to MiniCPM-o 2.6 model")
    # # parser.add_argument("--llama32_vision_11b_model_path", type=str, default=None, help="Path to Llama-3.2 11B Vision model")
    # # parser.add_argument("--llama32_vision_90b_model_path", type=str, default=None, help="Path to Llama-3.2 90B Vision model")


    # # --- OpenAI Arguments ---
    # parser.add_argument("--openai_api_key", type=str, default="sk-GYxssRBQRB9iPemjBpD5qfPSmlau4UXDG3nmNocEuvXqtvnc", help="OpenAI API Key (or set OPENAI_API_KEY env var)") 
    # parser.add_argument("--openai_base_url", type=str, default='https://boyuerichdata.chatgptten.com/v1', help="OpenAI API Base URL")
    
    # # --- Teacher-Student Search Arguments ---
    # parser.add_argument("--teacher_models", nargs='+', type=str, default=['qwen25_vl_72b', 'gemini-2.0-flash-thinking-exp-01-21'], help="List of teacher model names (must match keys in model_dict or be 'gpt-...' style)")
    # parser.add_argument("--student_models", nargs='+', type=str, default=['qwen25_vl_7b', 'qwen2_vl_7b'], help="List of student model names (must match keys in model_dict)")
    # parser.add_argument("--evaluator_model", type=str, default='chatgpt-4o-latest', help="Evaluator model name (must be GPT for default JUDGE_PROMPT)")
    # parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of reasoning steps to generate.")
    # parser.add_argument("--temperature1", type=float, default=1.2, help="First temperature for student model evaluation and teacher generation.")
    # parser.add_argument("--temperature2", type=float, default=0.3, help="Second temperature for student model evaluation.")
  

    # args = parser.parse_args()



    # 手动设置参数
    args_dict = {
        # 数据路径相关参数
        'data_path': '../rp_stage3_data/d2d_qa_plan_a_5.jsonl',  
        'image_dir_path': '../rp_stage3_data/images',      
        'output_path': '../output/tsis_results_d2d_5.jsonl',     
        'num_chunks': 1,                                  
        'chunk_idx': 0,                                   
    
        # 模型路径相关参数
        'qwen25_vl_72b_model_path': '',  #  /mnt/petrelfs/sunhaoran1/Radio--MCTS/model/Qwen2.5-VL-72B-Instruct-AWQ
        'qwen25_vl_7b_model_path': '/mnt/petrelfs/sunhaoran1/Radio--MCTS/model/Qwen2.5-VL-7B-Instruct',   
        'qwen2_vl_7b_model_path': '/mnt/petrelfs/sunhaoran1/Radio--MCTS/model/Qwen2-VL-7B-Instruct',      
        'internvl3_78b_model_path': '',           #   /mnt/hwfile/smart_health/jiangyankai/pretrained/InternVL3-78B
        'internvl3_8b_model_path': 'OpenGVLab/InternVL3-8B-Instruct',      # OpenGVLab/InternVL3-8B                                       
        'internvl3_9b_model_path': '',            # OpenGVLab/InternVL3-9B                               
        'minicpm_o_2_6_model_path': '',                                             #  openbmb/MiniCPM-o-2_6
        # 'llama32_vision_11b_model_path': None,                                                      
        # 'llama32_vision_90b_model_path': None,                                                      
    
        # OpenAI 相关参数
        'openai_api_key': 'sk-Bal815a7fc887ac7eefd144d59029530fd84f4de6e7vnUb3',                  # sk-ErhkQhqyFHbHREoHnL0qUxbRaY8uDC3nf7bdvpyT2zofb2Iq
        'openai_base_url': 'https://api.gptsapi.net/v1',                      #           http://35.220.164.252:3888/v1
        'qwen_api_key': 'sk-b639c6b0db4c4faf81ce919d37917240',                        
        'qwen_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'ds_api_key': 'sk-b5939300990144c9b2598206ec88c701',                        
        'ds_base_url': 'https://api.deepseek.com',
        'gemini_api_key': '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',
        'gemini_base_url': 'https://openrouter.ai/api/v1',
    
        # 教师-学生搜索相关参数
        'teacher_models': ['chatgpt-4o-latest', 'google/gemini-2.5-pro-preview-03-25'],          #  , 'google/gemini-2.5-pro-preview-03-25' 'qwen2.5-vl-72b-instruct'           # 教师模型名称列表
        'student_models': ['qwen25_vl_7b', 'qwen2_vl_7b', 'internvl3_8b'],                       # 'internvl3_8b', 'qwen2_vl_7b'                        # 学生模型名称列表
        'evaluator_model': 'deepseek-chat',                                                        
        'max_depth': 3,                                                                               
        'temperature1': 1.2,                                                                            
        'temperature2': 0.2}                                                                         

    args = argparse.Namespace(**args_dict)       

    errors = []
    model_name_to_arg_map = {
        'qwen25_vl_72b': 'qwen25_vl_72b_model_path',
        'internvl3_78b': 'internvl3_78b_model_path',
        'qwen25_vl_7b': 'qwen25_vl_7b_model_path',
        'internvl3_8b': 'internvl3_8b_model_path',
        'minicpm_o_2_6': 'minicpm_o_2_6_model_path',
        'qwen2_vl_7b': 'qwen2_vl_7b_model_path',
        'internvl3_9b': 'internvl3_9b_model_path',
        # Add any other mappings if needed
    }

    # Validate Teachers

    if len(args.teacher_models) < 2:
        errors.append("At least two teacher models must be specified via --teacher_models.")
    else:
        for model_name in args.teacher_models:
            if  'gpt' not in model_name and 'claude' not in model_name and 'gemini' not in model_name and '72' not in model_name: 
                arg_name = model_name_to_arg_map.get(model_name)
                if not arg_name:
                    errors.append(f"Teacher model '{model_name}' is not recognized or mapped to a path argument.")
                elif getattr(args, arg_name, None) is None:
                    errors.append(f"Path for teacher model '{model_name}' (expected argument --{arg_name}) was not provided.")

    # Validate Students
    if len(args.student_models) < 1:
        errors.append("At least one student model must be specified via --student_models.")
    else:
        for model_name in args.student_models:
            arg_name = model_name_to_arg_map.get(model_name)
            if not arg_name:
                 errors.append(f"Student model '{model_name}' is not recognized or mapped to a path argument.")
            elif getattr(args, arg_name, None) is None:
                 errors.append(f"Path for student model '{model_name}' (expected argument --{arg_name}) was not provided.")

    # Report errors if any
    # if errors:
    #     print("\nArgument Validation Errors:")
    #     for error in errors:
    #         print(f"- {error}")
    #     parser.print_help() 
    #     exit(1) 
    

    # --- Call the main inference function ---
    start_tsis(args)

