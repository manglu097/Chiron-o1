from openai import OpenAI
import json
from utils import (
    encode_image, locate_img, get_correctness,
    gpt_forward, replace_image_references, qwenplus_forward, ds_forward, extract_first_two_steps,
    select_best_mentor, process_case_info, select_next_step, remove_phrases, extract_first_step
)
from qwenvl_forward import qwenvl_forward
from internvl_forward import internvl_forward
from prompt import EVALUATE_PROMPT, JUDGE_PROMPT, REASONING_PROMPT
from step import Step
import time

class MentorInternSearch:
    def __init__(self, args):
        """
        Initialize the Mentor-Intern Search process.
        Args:
            args: Command line arguments containing model lists, paths, max_depth, etc.
        """
        
        self.root = Step(prefix_steps="") 
        self.args = args
        self.mentor_models = args.mentor_models
        self.intern_models = args.intern_models
        self.evaluator_model = args.evaluator_model 
        self.temperature1 = args.temperature1
        self.temperature2 = args.temperature2
        self.client1 = OpenAI(base_url=args.openai_base_url, api_key=args.openai_api_key)
        self.client2 = OpenAI(base_url=args.qwen_base_url, api_key=args.qwen_api_key)
        self.client3 = OpenAI(base_url=args.ds_base_url, api_key=args.ds_api_key)
        self.client4 = OpenAI(base_url=args.gemini_base_url, api_key=args.gemini_api_key)
        self.model_set = None 
        

    def _call_model_forward(self, model_name, prompt, temperature, base64_images=None, img_paths=None, extensions=None):
        
        try:
            if 'gpt' in model_name:
                print("Calling gpt model!")
                return gpt_forward(self.client1, prompt, base64_images, temperature=temperature, model_name=model_name, extensions=extensions)
            elif 'gemini' in model_name:
                print("Calling gemini model!")
                return gpt_forward(self.client4, prompt, base64_images, temperature=temperature, model_name=model_name, extensions=extensions)
            elif '72' in model_name:
                print("Calling qwen model!")
                return qwenplus_forward(self.client2, prompt, base64_images, temperature=temperature, model_name=model_name, extensions=extensions)
            elif 'qwen' in model_name: 
                 response = qwenvl_forward(
                     self.model_set[model_name]['model'], 
                     self.model_set[model_name]['processor'],
                     prompt, 
                     img_paths, 
                     temperature=temperature 
                 )
                 return response
            elif 'internvl' in model_name: 
                 response = internvl_forward(
                      self.model_set[model_name]['model'], 
                      self.model_set[model_name]['processor'],
                      prompt, 
                      img_paths, 
                      temperature=temperature 
                 )
                 return response
            else:
                print(f"Warning: Model type for '{model_name}' not recognized for forwarding. Returning None.")
                return None
        except Exception as e:
            print(f"Error calling model {model_name} (Temp: {temperature}): {e}")
            return None


    def _generate_next_step_with_mentor(self, mentor_model_name, step, question, base64_images, img_paths, gt_answer, case_info, extensions):
        """
        Generates the next single reasoning step using a mentor model.
        Returns the full reasoning chain, including prefix, current step, and future steps.
        """
        next_step_number = step.depth + 1
        reasoning_prefix = step.get_full_reasoning()

        if 'gpt' in mentor_model_name  or 'gemini' in mentor_model_name or '72' in mentor_model_name:
            case_info = replace_image_references(case_info)
            prompt = REASONING_PROMPT.format(
                reasoning_prefix=reasoning_prefix,
                question=question,
                gt_answer=gt_answer,
                case_info=case_info
            )
        else:
            prompt = REASONING_PROMPT.format(
                reasoning_prefix=reasoning_prefix,
                question=question,
                gt_answer=gt_answer,
                case_info=case_info
            )

        print("Starting to call mentor model!")
        time1 = time.time()
        response = self._call_model_forward(
            model_name=mentor_model_name,
            prompt=prompt,
            temperature=self.temperature1, 
            base64_images=base64_images,
            img_paths=img_paths,
            extensions=extensions
        )
        print(f"Mentor model call completed, time taken: {time.time() - time1} seconds")

        if response:
            return reasoning_prefix, response
        else:
            print(f" Failed to get response from {mentor_model_name} at generating step {next_step_number}.")
            return None

    def _evaluate_step_with_interns(self, step_to_evaluate, question, gt_answer, img_paths, mentors_scores):
        """Evaluates a generated step using intern models, each with two temperatures."""

        print(f"Evaluating step (Depth {step_to_evaluate.depth}, Mentor: {step_to_evaluate.generated_by}) with interns...")
        reasoning_prefix = step_to_evaluate.get_full_reasoning()
        correct_count = 0
        num_intern_models = len(self.intern_models)
        total_evaluations = num_intern_models * 2 

        if num_intern_models == 0:
            print("Warning: No intern models specified for evaluation. Returning score 0.")
            return 0.0
        
        for intern_model_name in self.intern_models:

            if 'internvl' in intern_model_name:
                image_prefix = ""
                for i, img_path in enumerate(img_paths, 1):
                    image_prefix += f"Image {i}: <image>\n"
                
                question = image_prefix + question

                intern_prompt = EVALUATE_PROMPT.format(
                question=question,
                reasoning_prefix=reasoning_prefix
                )
            else :
                intern_prompt = EVALUATE_PROMPT.format(
                question=question,
                reasoning_prefix=reasoning_prefix
                )
            
            for temp in [self.temperature1, self.temperature2]:
                print(f"  Running intern: {intern_model_name} (Temp: {temp})")
                time1 = time.time()
                
                intern_response = self._call_model_forward(
                    model_name=intern_model_name,
                    prompt=intern_prompt,
                    temperature=temp, 
                    img_paths=img_paths
                )
                print(f"Intern model call completed, time taken: {time.time() - time1} seconds")

                if not intern_response:
                    print(f"Failed to get response from intern {intern_model_name} (Temp: {temp})")
                    continue 

                if '### The final answer is:' not in intern_response:
                    print(f"Warning: Intern {intern_model_name} (Temp: {temp}) response did not contain '### The final answer is:'.")
                    continue

                model_answer = intern_response.split('### The final answer is:')[-1].strip()
                if not model_answer:
                     print(f"Warning: Intern {intern_model_name} (Temp: {temp}) provided empty final answer.")
                     continue

                judge_prompt = JUDGE_PROMPT.format(
                    question=question,
                    model_answer=model_answer,
                    gt_answer=gt_answer
                )
                
                time2 = time.time()
                judge_output = ds_forward(self.client3, judge_prompt, temperature=0.9, model_name=self.evaluator_model)
                print(f"Evaluator model call completed, time taken: {time.time() - time2} seconds")
                if judge_output:
                    is_correct = get_correctness(judge_output)
                    if is_correct == 1:
                        correct_count += 1 
                else:
                    print(f" Failed to get judgment from evaluator {self.evaluator_model}")

        # Calculate score based on total evaluations (2 * number of intern models)
        score = correct_count / total_evaluations if total_evaluations > 0 else 0.0
        print(f"  Step Score (Correct/Total Evaluations): {correct_count}/{total_evaluations} = {score:.2f}")


        return score

    def search(self, data, model_set, search_file, failed_search_file):
        """Performs the mentor-intern iterative search."""
        self.model_set = model_set 

        # 1. Initialize from data
        question = data["messages"][0]['content']
        question = question.replace('**Question:**', ' ').strip() 
        
        case_info = process_case_info(data)

        gt_answer = data["messages"][1]['content']
        gt_answer = gt_answer.replace('**Correct Answer:**', ' ').strip()
        try:
            img_paths, extensions = locate_img(data, self.args) 
            base64_images = [encode_image(img_path) for img_path in img_paths]
        except ValueError as e:
            print(f"Error encoding image for data item: {e}. Skipping item.")
            failed_data = data.copy()
            failed_data['error'] = str(e)
            failed_search_file.write(json.dumps(failed_data) + "\n")
            failed_search_file.flush()
            return
        except FileNotFoundError as e:
             print(f"Error finding image for data item: {e}. Skipping item.")
             failed_data = data.copy()
             failed_data['error'] = str(e)
             failed_search_file.write(json.dumps(failed_data) + "\n")
             failed_search_file.flush()
             return

        print(f"\n--- Starting Search for Question ID: {data.get('rid', 'N/A')} ---")

        initial_text = "Let's think about how to solve this problem clearly and reasonably step by step."
        self.root.text = initial_text
        current_step = self.root

        mentors_scores = {mentor: [] for mentor in self.mentor_models}
        reasoning_chains = {} 
        previous_mentors = [] 

        # 2. Loop through depth
        for depth in range(self.args.max_depth):
            print(f"\n-- Depth {depth} (Generating Step {depth + 1}) --")

            generated_children_for_step = []
            full_score_mentors = []
            all_zero_score = True # Set to False to cancel the early stop mechanism of all zero scores to save API consumption
            
            for mentor_model in self.mentor_models:
                print(f"Generating step with mentor: {mentor_model}")
                if current_step.parent is None:
                    prefix_reasoning,  suffix_reasoning = self._generate_next_step_with_mentor(
                    mentor_model, current_step, question, base64_images, img_paths, gt_answer, case_info, extensions
                )
                    complete_reasoning = prefix_reasoning + "\n" + suffix_reasoning
                    reasoning_chains[mentor_model] = complete_reasoning  
                    
                    new_step = extract_first_two_steps(suffix_reasoning)    
                    new_step = "### " + new_step
                else: 
                    if current_step.generated_by == mentor_model:
                        try:
                            new_step = reasoning_chains[mentor_model].split("###")[depth + 2].strip()
                            new_step = "### " + new_step
                        except IndexError:
                            new_step = ""

                    else:
                        prefix_reasoning,  suffix_reasoning = self._generate_next_step_with_mentor(
                        mentor_model, current_step, question, base64_images, img_paths, gt_answer, case_info, extensions
                    )

                        complete_reasoning = prefix_reasoning + "\n" + suffix_reasoning
                        reasoning_chains[mentor_model] = complete_reasoning  

                        new_step = extract_first_step(suffix_reasoning)
                        new_step = "### " + new_step
                
                temp_step = Step(step_text=new_step, prefix_steps=current_step.text, parent=current_step, generated_by=mentor_model)
                score = self._evaluate_step_with_interns(
                    temp_step, question, gt_answer, img_paths, mentors_scores
                )
                mentors_scores[mentor_model].append(score)

                if score > 0:
                    all_zero_score = False
                actual_child = current_step.add_child_step(step_text=new_step, score=score, generated_by=mentor_model)
                if actual_child: 
                    generated_children_for_step.append(actual_child)
                    if score == 1.0:
                        full_score_mentors.append(mentor_model)  

            # If any mentor's step is full score, select best and use its reasoning chain
            if full_score_mentors:
                if len(full_score_mentors) == 1:
                    best_mentor = full_score_mentors[0]
                else:
                    best_mentor = select_best_mentor(mentors_scores, depth)
                print(f"Full score achieved by {len(full_score_mentors)} mentors. Selected mentor: {best_mentor}")
                full_reasoning = reasoning_chains[best_mentor]
                
                result_data = {
                    'rid': data.get('rid', 'N/A'),
                    'images': data['images'],
                    'question': question,
                    'gt_answer': gt_answer,
                    'reasoning': full_reasoning,
                    'scores': mentors_scores,  
                    'final_depth': str(depth + 1),  
                    'generated_by': previous_mentors + [best_mentor],
                    'search_id': '1'
                }
                
                search_file.write(json.dumps(result_data) + "\n")
                search_file.flush()
                print(f"Saved successful result with full score for {data.get('rid', 'N/A')}")
                return  

            # If all scores are zero, stop and record as failure
            if all_zero_score:
                print("All mentor scores are zero. Stopping search and recording as failure.")
                failed_data = data.copy()
                failed_data['error'] = "All mentor scores are zero."
                failed_search_file.write(json.dumps(failed_data) + "\n")
                failed_search_file.flush()
                return

            # Select the best step to proceed
            if generated_children_for_step:
                current_step = select_next_step(generated_children_for_step, previous_mentors)
                previous_mentors.append(current_step.generated_by)
                print(f" Step (Depth {depth}) best child score: {current_step.score:.2f} (Mentor: {current_step.generated_by})")
            else:
                print("No valid steps generated. Stopping search.")
                break

            print("########################################################")

        # 3. Search finished (max depth reached)
        print(f"\n--- Search Completed (Max Depth Reached) ---")
        
        # 4. Save results - use the last selected mentor's reasoning
        if current_step.score >= 0:  
            final_mentor = current_step.generated_by
        else :
            final_mentor = "No valid mentor!"

        if final_mentor in reasoning_chains:
            final_reasoning = reasoning_chains[final_mentor]
            result_data = {
                'rid': data.get('rid', 'N/A'),
                'images': data['images'],
                'question': question,
                'gt_answer': gt_answer,
                'reasoning': final_reasoning,
                'scores': mentors_scores,  
                'final_depth': str(depth + 1),  
                'generated_by': previous_mentors,  # {current_step.get_step_path()[i].generated_by for i in range(len(current_step.get_step_path()))}
                'search_id': '0'
            }
                    
            search_file.write(json.dumps(result_data) + "\n")
            search_file.flush()
            print(f"Saved result for {data.get('rid', 'N/A')}")
        else:
            failed_data = data.copy()
            failed_data['error'] = "Search failed to find a valid reasoning path."
            failed_search_file.write(json.dumps(failed_data) + "\n")
            failed_search_file.flush()
            print(f"Saved failed search log for {data.get('rid', 'N/A')}")

        print(f"--- Search Finished for Question ID: {data.get('rid', 'N/A')} ---")