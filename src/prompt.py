JUDGE_PROMPT = """
Evaluate whether the model's answer aligns with the correct answer semantically. Output 'Yes' only if the model's answer matches the correct result, and 'No' if it does not match or if the correctness is unclear. Provide only 'Yes' or 'No' as the output, without any explanation.

Question: {question}
Model's answer: {model_answer}
Correct answer: {gt_answer}"""




REASONING_PROMPT = """
Given a specific question about the images, the patient's case information (such as age, gender, chief complaint and some relevant image analysis ), your goal is to generate a detailed, step-by-step thought process that leads to the correct answer.
1. Your thought process must rely solely on the provided information. Do not fabricate details or introduce information not present in the inputs.
2. Approach the task as if the answer is unknown, avoiding any shortcuts or assumptions that the gold standard answer is already understood. 
3. If the thought process involves observations related to images, present those observations as if they were directly derived from the images themselves, without referencing image analysis.
4. Adapt your thought process to the complexity of each case, using fewer reasoning steps for simpler problems and more thorough analysis for complex ones, mirroring the flexible and analytical mindset of a skilled clinician.

Format your response with the following format:
### Step 1: 
### Step 2:
...
### The final answer is: 

Case Information:
    {case_info}
Question: {question}
Correct Answer: {gt_answer}
Reasoning prefix: {reasoning_prefix}

Continue your response by building directly on the existing thought process, ensuring that no changes are made to the content of the prior reasoning, and provide a clear and coherent completion of your answer. Just output the completed part.
"""


EVALUATE_PROMPT = """
Using the provided medical images and partial thought process, deduce the correct answer of the question through rigorous reasoning. Ensure the response is concise, accurate, and conforms to medical terminology standards. Provide only the final answer.

Format your response with the following format:
### The final answer is: 

Question: {question}
Reasoning prefix: {reasoning_prefix}
"""