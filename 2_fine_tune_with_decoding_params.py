from llama_cpp import Llama

llm = Llama(
    model_path="Meta-Llama-3-8B.Q2_K.gguf",
)

concise_output = llm(
    "What is the most used database for data storage?",
    temperature=0.2,
    top_k=1,
    top_p=0.4,
    max_tokens=100,
)

print(concise_output["choices"][0]["text"])

"""
1. temperature
=================================================================
Controls the randomness of the output.

Range: 0.0 to 2.0

Default: ~0.8

Effect:

🔥 Higher (>1.0): More creative, risky, surprising outputs.

❄️ Lower (<0.5): More focused, deterministic, and repetitive.

0.0: Deterministic — always gives the same output.

You used 0.2 — very conservative. Great for concise, factual answers.

2. top_k
=================================================================
Limits sampling to the top K most likely tokens at each step.

Range: 1 to 1000+

Default: ~40 or 50

Effect:

top_k=1: Only pick the most likely next word (very deterministic).

top_k=50: Consider top 50 most likely options — more varied.

top_k=0: Disables top-k filtering.

You used top_k=1 — it forces selection of the single most likely token. Useful when you want extremely concise and predictable responses.

3. top_p (a.k.a. nucleus sampling)
=================================================================
Controls diversity by focusing on the smallest cumulative probability mass p.

Range: 0.0 to 1.0

Default: ~0.9

Effect:

top_p=1.0: No filtering.

top_p=0.9: Keep sampling from the smallest set of tokens whose total probability ≥ 90%.

Lower values = more conservative, focused output.

You used top_p=0.4 — only a small probability range is considered, which increases precision but can limit creativity.

4. max_tokens
=================================================================
Specifies the maximum number of tokens to generate in the response.

Range: Any positive integer up to the model’s context limit (e.g., 2048, 4096, 8192, or even 128k for some LLaMA variants).

Default: Varies by library — often 512 or None (unbounded).

Effect:

Controls the length of the output.

Acts as a hard cutoff (prevents runaway responses).

You used max_tokens=100 — that’s usually enough for a concise answer.
"""