"""
 create messages to prompt a customer support chatbot for an internet service provider.
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="medmekk/Llama-3.2-1B-Instruct.GGUF",
    filename="Llama-3.2-1B-Instruct-IQ3_M_imat.gguf",
    n_gpu_layers=-1  # Set the number of GPU layers to -1 for automatic detection
)

prompt = "Give me four short steps to troubleshoot my internet connection."

conv = [
	# Complete the user message
	{
        "role": "user",
	    "content": prompt
    }
]

# Pass the conversation to the model
result = llm.create_chat_completion(messages=conv, max_tokens=200)
print(result['choices'][0]['message']['content'])