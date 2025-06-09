from qwen_vl_utils import process_vision_info

def qwenvl_forward(model, processor, prompt, img_paths, temperature=0.9):
    messages = [
        {
            'role': "system",
            "content": 'You are a highly professional and experienced clinician. You are familiar with the latest medical guidelines, diagnostic standards and treatment plans, and can reasonably answer users\' questions.'
        },
        {
            "role": "user",
            "content": [],
        },
    ]
    
    for img_path in img_paths:
        messages[1]["content"].append({
            "type": "image",
            "image": img_path,
        })
    
    messages[1]["content"].append({"type": "text", "text": prompt})

    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)  
    inputs = processor(
        text=[texts],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=2048, repetition_penalty=1, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_texts
