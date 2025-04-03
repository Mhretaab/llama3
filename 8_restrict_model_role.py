"""
utilize a system message to steer the model into only answering customer questions about their internet service.
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="medmekk/Llama-3.2-1B-Instruct.GGUF",
    filename="Llama-3.2-1B-Instruct-IQ3_M_imat.gguf",
    n_gpu_layers=-1  # Set the number of GPU layers to -1 for automatic detection
)

# Add a system message to the conversation list
conv = [
    {
        "role": "system",
        "content": "You are a helpful and professional customer support assistant for an internet service provider. If the question or instruction doesn't relate to internet service, quote the response: 'Sorry, I can't answer that.'"
    },
    {
        "role": "user",
        "content": "Help me decide which stocks to invest in."
    }
]

result = llm.create_chat_completion(messages=conv, max_tokens=200)
# Extract the model response from the result object
assistant_content = result["choices"][0]["message"]["content"]
print(assistant_content)
