
#This imports the Llama class from the llama_cpp Python package (which is a Python wrapper around llama.cpp, a fast inference engine for LLaMA models).
from llama_cpp import Llama



#Initialize the LLaMA model from local GGUF
# This line loads a quantized LLaMA 3 model (in .gguf format) from a local file.

# "Meta-Llama-3-8B.Q2_K.gguf" is the path to the model file.

# No weights are downloaded — the file must already be on disk.

# This is faster than using .from_pretrained() which downloads from Hugging Face.
llm = Llama(
    model_path="Meta-Llama-3-8B.Q2_K.gguf",
)

#Define a question prompt

question = "What is the most used database for data storage?"

#Run inference
#This sends the question to the LLaMA model, which generates a response.
#This is a completion-style call (not chat-style). It expects a prompt and returns the continuation (like GPT-3's text-davinci-003).
response = llm(question)
print(response["choices"][0]["text"])


# llm = Llama.from_pretrained(
#     repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
#     filename="Meta-Llama-3-8B.Q2_K.gguf",
#     n_gpu_layers=-1
# )

# output = llm(
#     "What are the different ways to improve kotlin code?"
# )