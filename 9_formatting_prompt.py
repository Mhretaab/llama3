"""
Models can sometimes struggle to separate the task, expected output, and additional context from a long, unstructured prompt. 
To remedy this, you can insert clear labels to break up and differentiate this information for the model.

Add the labels Instruction, Question, and Answer to the prompt to format it more effectively.
Pass the prompt to the model.
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="medmekk/Llama-3.2-1B-Instruct.GGUF",
    filename="Llama-3.2-1B-Instruct-IQ3_M_imat.gguf",
    n_gpu_layers=-1  # Set the number of GPU layers to -1 for automatic detection
)

# Add formatting to the prompt
prompt = """
Instruction: Explain the concept of gravity in simple terms.
Question: What is gravity?
Answer:
"""

#  Send the prompt to the model
output = llm(prompt, max_tokens=250, stop=["Question:"])
print(output['choices'][0]['text'])
