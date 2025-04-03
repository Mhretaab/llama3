"""
You work for an e-commerce company and are integrating Llama into a customer support assistant. 
The assistant answers frequently asked questions, but you've noticed that responses are too repetitive.
You need to modify decoding parameters to encourage more varied wording while keeping responses informative.
"""

from llama_cpp import Llama

llm = Llama(
    model_path="Meta-Llama-3-8B.Q2_K.gguf",
    n_gpu_layers=-1, # Set the number of GPU layers to -1 for automatic detection
)

output = llm(
    "Can I exchange an item I purchased?",
    # Set the temperature parameter to provide more varied responses
    temperature=1,
    max_tokens=100
)

print(output["choices"][0]["text"])
