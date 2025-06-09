import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import json
from bert_score import score
from openai import OpenAI
from tqdm import tqdm
import argparse

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def ds_forward(client, pred_answer, ground_truth, temperature=0.9):
    message = [
        {"role": "user", "content": f"Evaluate whether the model's answer is semantically similar to the correct answer. Output 'Yes' if the model's answer conveys a similar meaning to the correct result, even if the wording differs, and 'No' if it does not. Provide only 'Yes' or 'No' as the output, without any explanation.\nModel's answer: {pred_answer}\nCorrect answer: {ground_truth}"}
    ]
    completion = client.chat.completions.create(
        model='deepseek-chat',
        messages=message,
        temperature=temperature
    )
    return completion.choices[0].message.content.strip()

def get_correctness(judge_output):
    if 'yes' in judge_output.lower():  
        return 1
    else:
        return -1

def evaluate_model(vqa_json_path, image_dir, model_path, output_path, api_key):
    client = OpenAI(base_url='https://api.deepseek.com',  api_key=api_key)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    with open(vqa_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Initialize metrics
    closed_correct = 0
    closed_total = 0
    open_bertscore_sum = 0.0
    open_count = 0
    results_output = []

    # Create or clear the output file
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[]")  
    for idx, entry in enumerate(tqdm(data, desc="Processing Entries", unit="entry")):
        id = entry["id"]
        image_names = entry["img_name"]  
        question = entry["question"]
        ground_truth = entry["answer"]
        answer_type = entry["answer_type"]
        if isinstance(image_names, str):  
            image_names = [image_names]
        pixel_values_list = []
        num_patches_list = []

        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            pixel_values_single = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values_single)
            num_patches_list.append(pixel_values_single.size(0))
        pixel_values = torch.cat(pixel_values_list, dim=0)
        # Construct prompt
        prompt = f"{question}"
        # Call the model for inference
        response = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list  
        )
        # Get predicted answer
        pred_answer = response.strip()
        if "### The final answer is:" not in pred_answer:
            continue
        reason_answer = pred_answer.split("### The final answer is:")[0].strip()
        pred_answer = pred_answer.split("### The final answer is:")[1].strip()
        # Print question and model's output
        print(f"Question: {question}")
        print(f"Model Answer: {pred_answer}")
        print(f"Ground Truth: {ground_truth}")
        print("========================================")
        # Calculate accuracy
        closed_total += 1
        judge_output = ds_forward(client, pred_answer, ground_truth, temperature=0.9)
        if judge_output:
            is_correct = get_correctness(judge_output)
            if is_correct == 1:
                closed_correct += 1
        else:
            print(f"Failed to get judgment from evaluator")
        # Calculate BERTScore
        P, R, F1 = score(
            cands=[pred_answer],
            refs=[ground_truth],
            model_type="roberta-large",
            num_layers=17,
            idf=False,
            batch_size=4,
            verbose=True,
            lang="en"
        )
        f1_score = F1.item()
        open_bertscore_sum += f1_score
        open_count += 1
        # Construct a single result entry
        result_entry = {
            "index": id,
            "image_name": image_names,  
            "question": question,
            "reason_answer": reason_answer,
            "model_answer": pred_answer,
            "ground_truth": ground_truth,
            "answer_type": answer_type,
            "result": [closed_correct, closed_total, open_bertscore_sum, open_count]
        }
        # Append the result to the file
        with open(output_path, "r+", encoding="utf-8") as f:
            results = json.load(f)
            results.append(result_entry)
            f.seek(0)
            json.dump(results, f, ensure_ascii=False, indent=4)
    # Calculate final metrics
    closed_acc = closed_correct / closed_total if closed_total > 0 else 0
    open_bertscore_avg = open_bertscore_sum / open_count if open_count > 0 else 0
    print("====== Evaluation Results ======")
    print(f"CLOSED Questions Acc: {closed_acc:.4f}")
    print(f"OPEN   Questions BERTScore-F1: {open_bertscore_avg:.4f}")
    # Append final metrics to the file
    final_metrics = {
        "CLOSED Questions Acc": closed_acc,
        "OPEN   Questions BERTScore-F1": open_bertscore_avg
    }
    with open(output_path, "r+", encoding="utf-8") as f:
        results = json.load(f)
        results.append(final_metrics)
        f.seek(0)
        json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    """
    Main function to evaluate multiple models.
    """
    parser = argparse.ArgumentParser(description="Evaluate models on VQA tasks.")
    parser.add_argument("--vqa_json_path", type=str, required=True, help="Path to the VQA JSON file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the OpenAI client.")
    args = parser.parse_args()

    vqa_json_path = args.vqa_json_path
    image_dir = args.image_dir
    model_path = args.model_path
    output_path = args.output_path
    api_key = args.api_key

    print(f"Evaluating model: {model_path}")
    evaluate_model(vqa_json_path, image_dir, model_path, output_path, api_key)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()