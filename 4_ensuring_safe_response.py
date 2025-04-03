"""
You're configuring an internal chatbot for a medical team. To ensure consistent responses, you need to limit variability 
by setting a token limit and restricting token selection.

You have been provided the Llama class instance in the llm variable and the code to call the completion. You are also given a sample 
prompt to test with.
"""

from llama_cpp import Llama

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="medmekk/Llama-3.2-1B-Instruct.GGUF",
	filename="Llama-3.2-1B-Instruct-IQ3_M_imat.gguf",
    n_gpu_layers=-1 # Set the number of GPU layers to -1 for automatic detection
)

output = llm(
		"What are the symptoms of strep throat?", 
  		# Set the model parameters 
      	max_tokens=100, #Limit response length
		top_k=2 #Restrict word choices
) 

print(output['choices'][0]['text'])
