import json
import base64
import os
import math
from PIL import Image
import io
import math
import random
import re

def resize_image_if_needed(image_path, max_dimension=512):
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            
            if max(width, height) > max_dimension:
                scale_factor = max_dimension / max(width, height)
                new_width = int(round(width * scale_factor))
                new_height = int(round(height * scale_factor))
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image.copy()
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def encode_to_base64(buffered):
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image(image_path):
    try:
        with Image.open(image_path) as image:
            resized_image = resize_image_if_needed(image_path)
            original_format = image.format or 'PNG'
            
            buffered = io.BytesIO()
            resized_image.save(buffered, format=original_format)
            
            return encode_to_base64(buffered)
    except Exception as e:
        print(f"Error encoding image: {e}")
        raise


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)  
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def locate_img(d, args):
    valid_paths = []
    extensions = []
    for img in d['images']:
        if os.path.exists(os.path.join(args.image_dir_path, img)):
            full_path = os.path.join(args.image_dir_path, img)
            valid_paths.append(full_path)
            extensions.append(os.path.splitext(img)[1][1:].lower())
        elif os.path.exists(img):
            valid_paths.append(img)
            extensions.append(os.path.splitext(img)[1][1:].lower())
    if not valid_paths:
        raise ValueError(f"No valid image paths found in: {d['image']}")
    return valid_paths, extensions



def gpt_forward(client, prompt, base64_images=None, temperature=0.9, model_name="chatgpt-4o-latest", extensions=None):
    content = [{
                    "type": "text",
                    "text": prompt
                }]
    if base64_images is not None and extensions is not None:
        for base64_image, extension in zip(base64_images, extensions):
            content.append({
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/{extension};base64,{base64_image}"
                }
            })
    message = [
            {"role": "system", "content": """You are a highly professional clinician with expertise across multiple clinical fields, 
             well-versed in the latest medical guidelines, diagnostic standards, and treatment plans, capable of providing accurate and reasoned responses to user inquiries."""},
            {"role": "user", "content": content}
        ]
    completion = client.chat.completions.create(
        model = model_name,
        messages = message,
        temperature = temperature
    )

    return completion.choices[0].message.content


def qwenplus_forward(client, prompt, base64_images=None, temperature=0.9, model_name="qwen-vl-plus", extensions=None):
    content = [{
                    "type": "text",
                    "text": prompt
                }]
    if base64_images is not None and extensions is not None:
        for base64_image, extension in zip(base64_images, extensions):
            content.append({
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/{extension};base64,{base64_image}"
                }
            })
    message = [
            {"role": "system", "content": """You are a highly professional clinician with expertise across multiple clinical fields, 
             well-versed in the latest medical guidelines, diagnostic standards, and treatment plans, capable of providing accurate and reasoned responses to user inquiries."""},
            {"role": "user", "content": content}
        ]
    completion = client.chat.completions.create(
        model = model_name,
        messages = message,
        temperature = temperature
    )

    return completion.choices[0].message.content


def ds_forward(client, prompt, temperature=0.9, model_name="chatgpt-4o-latest"):
    message = [
            {"role": "system", "content": """You are a highly professional clinician with expertise across multiple clinical fields, 
             well-versed in the latest medical guidelines, diagnostic standards, and treatment plans, capable of providing accurate and reasoned responses to user inquiries."""},
            {"role": "user", "content": prompt}
        ]
    completion = client.chat.completions.create(
        model = model_name,
        messages = message,
        temperature = temperature
    )

    return completion.choices[0].message.content.strip()


def get_correctness(judge_output):
    if 'yes' in judge_output.lower():  
        return 1
    else:
        return -1


def select_best_mentor(mentors_scores, current_depth):
    """
    Select the best mentor based on their past scores.
    If at the first step, randomly select a mentor.
    If beyond the first step, calculate the competitiveness score for each mentor
    as the product of their past scores and select the one with the highest score.
    If scores are tied, randomly select among them.

    Args:
        mentors_scores (dict): A dictionary where keys are mentor names and values are lists of scores.
        current_depth (int): The current depth of exploration.

    Returns:
        str: The name of the selected mentor.
    """
    if current_depth == 0:
        # Randomly select a mentor if at the first step
        return random.choice(list(mentors_scores.keys()))

    # Calculate competitiveness score for each mentor
    competitiveness_scores = {}
    for mentor, scores in mentors_scores.items():
        competitiveness_scores[mentor] = 1
        for score in scores:
            competitiveness_scores[mentor] += score

    # Find the maximum competitiveness score
    max_score = max(competitiveness_scores.values())
    best_mentors = [mentor for mentor, score in competitiveness_scores.items() if score == max_score]

    # Randomly select among the best mentors if there's a tie
    return random.choice(best_mentors)


def process_case_info(data):
    """
    Process the case information and format it into a structured string.

    Args:
        data (dict): A dictionary containing case information.

    Returns:
        str: A formatted string with presentation, age, gender, and caption details.
    """
   
    presentation = data.get("presentation", "N/A")
    age = data.get("age_label", "N/A")
    gender = data.get("gender_label", "N/A")
    captions = data.get("caption", [])

    
    formatted_string = f"Chief complaint: {presentation}\n"
    formatted_string += f"Age: {age}\n"
    formatted_string += f"Gender: {gender}\n"

    if captions:
        formatted_string += "Image analysis:\n"
        for idx, caption in enumerate(captions, start=1):
            formatted_string += f"Modality {idx}: {caption}\n"
    else:
        formatted_string += "Captions: None\n"

    return formatted_string

def select_next_step(steps, previous_mentors=None):
    """
    Select the next reasoning step. If multiple steps have the same score, 
    prioritize steps generated by mentors that have not been selected before.
    
    Args:
        steps (list): List of steps, each with score and generated_by attributes
        previous_mentors (list, optional): List of previously selected mentor models, default is None
        
    Returns:
        Step: The selected next step
    """
    if not steps:
        return None
    
    max_score = max(s.score for s in steps)
    top_steps = [s for s in steps if s.score == max_score]
    
    # If there's only one highest scoring step, return it directly
    if len(top_steps) == 1:
        return top_steps[0]
    
    # When scores are tied, find steps generated by mentors not previously selected
    new_mentors_steps = [s for s in top_steps if s.generated_by not in previous_mentors]

    if new_mentors_steps:
        return random.choice(new_mentors_steps)
    
    return random.choice(top_steps)


def replace_image_references(text):
    """
    Replace all occurrences of 'According to the <image>\n<image>\n...' pattern with 
    'According to the image 1, image 2,...' where numbers are continuously incremented
    throughout the whole text.

    Args:
        text (str): The input string containing the phrases.

    Returns:
        str: The modified string with the phrases replaced.
    """
    image_counter = 0
    
    def replace_with_numbered_images(match):
        nonlocal image_counter
        image_count = match.group(0).count('<image>')
        
        if image_count == 0:
            return match.group(0)
        
        image_numbers = range(image_counter + 1, image_counter + image_count + 1)
        image_counter += image_count
        
        numbered_images = ", ".join([f"image {i}" for i in image_numbers])
        
        if match.group(0).rstrip().endswith(","):
            return f"According to the {numbered_images}, "
        else:
            return f"According to the {numbered_images} "
    
    pattern = r"According to the (?:<image>\n)+,?"
    replaced_text = re.sub(pattern, replace_with_numbered_images, text)
    return replaced_text

def extract_first_step(text):
    """
    Extract the content of the first step from the input text.

    Args:
        text (str): The input string containing multiple steps.

    Returns:
        str: The content of the first step, or an empty string if not found.
    """
    
    parts = text.split("###")
    
    for part in parts:
        if "Step" in part:
            return part.strip()

    return ""

def extract_first_two_steps(text):
    """
    Extract the content of the first two steps from the input text.

    Args:
        text (str): The input string containing multiple steps.

    Returns:
        str: The content of the first two steps, or an empty string if not found.
    """
    parts = text.split("###")
    
    steps = []
    for part in parts:
        if "Step" in part and len(steps) < 2:
            steps.append(part.strip())
    
    if len(steps) == 0:
        return ""
    elif len(steps) == 1:
        return steps[0]
    else:
        return steps[0] + "\n\n### " + steps[1]

def remove_phrases(text):
    """
    Remove all occurrences of 'According to the <image>\n,' and 
    'According to the <image>\n<image>\n,' from the input text.

    Args:
        text (str): The input string containing the phrases.

    Returns:
        str: The modified string with the phrases removed.
    """
    cleaned_text = re.sub(r"According to the (<image>\n)+,", "", text)
    return cleaned_text
