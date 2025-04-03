"""
Creating a JSON inventory list
Being able to use LLMs to generate structured outputs means that they can be used to reliably exchange information 
between systems with minimal processing.

You'll test your prompting skills by converting a stock inventory list written in natural language into JSON format. 
A model and system prompt have already been defined for you to use.
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
	filename="Llama-3.2-1B-Instruct-IQ3_M.gguf"
)

output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You convert inventory lists from text to JSON, extracting item counts and names from the text as keys and values in the form: item: count; for example, 'banana': 32.",},
            {"role": "user", "content": "Fifteen apples, thirty-three oranges, and five thousand fifty-two potatoes."},
        ],
		# Specify output format to JSON
        response_format={
            "type": "json_object",
        }
)

print(output['choices'][0]['message']['content'])