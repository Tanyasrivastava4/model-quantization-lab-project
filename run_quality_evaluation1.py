import subprocess
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
MODELS = {
    "FP16": "models/llama-2-7b-fp16.gguf",
    "Q4_K_M": "models/llama-2-7b-q4_K_M.gguf"
}

LLAMA_BIN = "llama.cpp/build/bin/llama-run"  # Make sure this path is correct
LOG_PATH = Path("logs/quality_benchmark.log")
os.makedirs(LOG_PATH.parent, exist_ok=True)

# -----------------------------
# Prompts (20 diverse)
# -----------------------------
PROMPTS = [
    "Explain AI in simple words.",
    "Summarize Python programming language.",
    "Write a short story about a robot.",
    "Translate 'Hello, how are you?' to French.",
    "What is quantum computing?",
    "Explain the importance of data privacy.",
    "Describe how a blockchain works.",
    "Give an example of recursion in Python.",
    "Explain Newton's laws of motion.",
    "Write a haiku about winter.",
    "Describe the process of photosynthesis.",
    "Summarize the plot of Romeo and Juliet.",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a SQL query to find top 5 sales.",
    "Explain the theory of relativity.",
    "Describe the life cycle of a butterfly.",
    "Generate a motivational quote.",
    "Explain what a neural network is.",
    "Give a recipe for chocolate chip cookies.",
    "Explain climate change in simple words."
]

# -----------------------------
# Run models and collect outputs
# -----------------------------
results = {"FP16": {}, "Q4_K_M": {}}
timings = {"FP16": {}, "Q4_K_M": {}}

with open(LOG_PATH, "w") as log_file:
    log_file.write("Quality Benchmark Log\n")
    log_file.write("="*60 + "\n")

    for name, model_path in MODELS.items():
        log_file.write(f"Model: {name}\n")
        print(f"\nRunning {name} model...")
        for prompt in PROMPTS:
            print(f"Prompt: {prompt}")
            start = time.time()

            # -----------------------------
            # Adjust command for FP16 vs Q4_K_M
            # -----------------------------
            if "Q4" in name:  # Quantized models
                cmd = [
                    LLAMA_BIN,
                    model_path,
                    prompt,
                    "--threads", "4",
                    "--max-tokens", "256"
                ]
            else:  # FP16 models
                cmd = [
                    LLAMA_BIN,
                    "-m", model_path,
                    "-p", prompt,
                    "--threads", "4",
                    "--max-tokens", "256"
                ]

            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            except FileNotFoundError:
                raise Exception(f"Binary not found at {LLAMA_BIN}")
            except subprocess.CalledProcessError as e:
                print(f"Error running model {name} on prompt: {prompt}")
                print(e.stderr)
                results[name][prompt] = ""
                timings[name][prompt] = {"elapsed": 0, "tokens": 0, "tokens_per_sec": 0}
                continue

            elapsed = time.time() - start
            output = result.stdout
            # Clean llama-run output
            output = output.replace("<|im_start|>assistant", "").replace("<|im_end|>", "").strip()
            results[name][prompt] = output

            # Count tokens: simple split by space
            token_count = len(output.split())
            timings[name][prompt] = {
                "elapsed": elapsed,
                "tokens": token_count,
                "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0
            }

            log_file.write(f"Prompt: {prompt}\nTime: {elapsed:.2f}s\nTokens: {token_count}\nOutput:\n{output}\n{'-'*50}\n")
            print(f"Done in {elapsed:.2f}s, Tokens: {token_count}, Tokens/sec: {timings[name][prompt]['tokens_per_sec']:.2f}\n")

# -----------------------------
# Compute quality degradation using embeddings
# -----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
quality_report = []

for prompt in PROMPTS:
    fp16_output = results["FP16"].get(prompt, "")
    q4_output = results["Q4_K_M"].get(prompt, "")

    fp16_emb = embed_model.encode(fp16_output, convert_to_numpy=True) if fp16_output else None
    q4_emb = embed_model.encode(q4_output, convert_to_numpy=True) if q4_output else None
    sim_score = cosine_similarity([fp16_emb], [q4_emb])[0][0] if fp16_emb is not None and q4_emb is not None else 0.0

    quality_report.append({
        "prompt": prompt,
        "fp16_output": fp16_output,
        "q4_output": q4_output,
        "cosine_similarity": sim_score,
        "degradation_%": (1 - sim_score) * 100,
        "fp16_tokens": timings["FP16"].get(prompt, {}).get("tokens", 0),
        "q4_tokens": timings["Q4_K_M"].get(prompt, {}).get("tokens", 0),
        "fp16_tokens_sec": timings["FP16"].get(prompt, {}).get("tokens_per_sec", 0),
        "q4_tokens_sec": timings["Q4_K_M"].get(prompt, {}).get("tokens_per_sec", 0),
        "fp16_elapsed_sec": timings["FP16"].get(prompt, {}).get("elapsed", 0),
        "q4_elapsed_sec": timings["Q4_K_M"].get(prompt, {}).get("elapsed", 0)
    })

df = pd.DataFrame(quality_report)
df.to_csv("quality_report.csv", index=False)
print("Quality report saved to quality_report.csv")
