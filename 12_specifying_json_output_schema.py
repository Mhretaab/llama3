"""
You are part of a team working on an online education platform designing new interactive exercise where students are able to ask 
questions and their answer is displayed through a graphical view. This question-answering feature is powered by an LLM, but the 
graphical view requires a JSON input with the fields Question and Answer to correctly show the question and answer:
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-IQ3_M.gguf"
)

messages = [
    {"role": "system", "content": "You are a helpful tutor that answers questions from students. Please provide your responses in the following JSON format: {\"Question\": \"<user's question>\", \"Answer\": \"<your answer>\"}."},
    {"role": "user", "content": "is gravity a force according to Einstein?"},
]

output = llm.create_chat_completion(
    messages=messages,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            # Set the properties of the JSON fields and their data types
            "properties": {"Question": {"type": "string"}, "Answer": {"type": "string"}}
        }
    }
)

print(output['choices'][0]['message']['content'])
