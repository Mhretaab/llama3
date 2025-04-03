"""
You're developing an AI-powered content assistant for a SaaS marketing team. The team needs to automate social media posts about 
their latest software updates, and you need to adjust response diversity so that multiple calls to the model result in different 
variations.

Adjust the top-p parameter to a value in the upper half of its range so that it generates more varied responses.
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="medmekk/Llama-3.2-1B-Instruct.GGUF",
    filename="Llama-3.2-1B-Instruct-IQ3_M_imat.gguf",
    n_gpu_layers=-1  # Set the number of GPU layers to -1 for automatic detection
)

output = llm(
    "Write a tweet announcing a new analytics dashboard feature for enterprise users.",
    max_tokens=150,
    # Set top-p to a value in the upper range for more varied responses
    top_p=1
)

print(output['choices'][0]['text'])
