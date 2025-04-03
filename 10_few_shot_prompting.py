"""
Few-shot prompting with Llama
You're using a Llama model to identify the sentiment of customer reviews from Google and Yelp as Positive or Negative. 
To ensure these labels are consistent for each review, you'll design a few-shot prompt containing three examples.

Here are the examples you want to provide to the model:

I ordered from this place last night, and I'm impressed! → Positive
My order was delayed by over an hour without any updates. Disappointing! → Negative
The food quality is top-notch. Highly recommend! → Positive
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
	filename="Llama-3.2-1B-Instruct-IQ3_M.gguf",
)


# Complete the few-shot prompt
prompt = """
Review 1: I ordered from this place last night, and I'm impressed! 
Sentiment 1: Positive, 
Review 2: My order was delayed by over an hour without any updates. Disappointing!  
Sentiment 2: Negative,
Review 3: The food quality is top-notch. Highly recommend! 
Sentiment 3: Positive,
Review 4: Delicious food, and excellent customer service! 
Sentiment 4: 
"""

# Send the prompt to the model with a stop word
output = llm(
    prompt,
    max_tokens=5,
    stop=["Review"]
)

print(output["choices"][0]["text"])
