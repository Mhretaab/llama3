from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="medmekk/Llama-3.2-1B-Instruct.GGUF",
    filename="Llama-3.2-1B-Instruct-IQ3_M_imat.gguf",
    n_gpu_layers=-1  # Set the number of GPU layers to -1 for automatic detection
)

system_message = "You are a software engineer working as consultant for a talent acquisition company. You design and develop applications using state-of-the-art technoligies for its clients."
user_message = "Write a Java program that functions as mathematical calculator. It should be able to add, subtract, multiply and divide two numbers. The program should also be able to calculate the square root of a number. The program should be able to handle invalid inputs gracefully."

message_list = [
    {
        "role": "system",
        "content": system_message
    },
    {
        "role": "user",
        "content": user_message
    }
]

response = llm.create_chat_completion(
    messages=message_list,
    max_tokens=4096,
    temperature=0.7,
    top_p=1
)

print(response['choices'][0]['message']['content'])
