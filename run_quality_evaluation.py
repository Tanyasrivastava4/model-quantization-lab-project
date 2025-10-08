#import subprocess
#import csv
#from pathlib import Path

# Paths
#FP16_MODEL = Path("models/llama-2-7b-fp16.gguf")
#Q4_MODEL = Path("models/llama-2-7b-q4_K_M.gguf")
#LLAMA_BIN = Path("llama.cpp/build/bin/llama-run")
#OUTPUT_CSV = Path("quality_report.csv")

# List of 20 diverse prompts
#PROMPTS = [
 #   "Explain artificial intelligence in simple words.",
  #  "Summarize Python programming language.",
  #  "Write a short story about a robot.",
   # "Explain the concept of blockchain.",
   # "Give me a simple explanation of quantum computing.",
  #  "Describe the importance of data privacy.",
  #  "Write a poem about the ocean.",
  #  "Explain machine learning to a 10-year-old.",
  #  "What are the applications of AI in healthcare?",
  #  "Write a motivational quote.",
  #  "Explain climate change in simple terms.",
  #  "Describe the life cycle of a butterfly.",
  #  "Summarize the plot of Romeo and Juliet.",
  #  "Explain the difference between RAM and ROM.",
  #  "Give tips for effective time management.",
  #  "Describe the process of photosynthesis.",
  #  "Explain how a car engine works.",
  #  "Write a dialogue between two friends about AI.",
  #  "Explain the concept of neural networks.",
  #  "Summarize the history of the Internet."
#]

# Function to run a model and clean output
#def run_model(model_path, prompt, threads=4, max_tokens=256):
 #   cmd = [
  #      str(LLAMA_BIN),
   #     str(model_path),
    #    prompt,
     #   "--threads", str(threads),
      #  "--max-tokens", str(max_tokens)
    #]
   # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   # text = result.stdout.decode("utf-8", errors="ignore")
    # Remove chat tokens
  #  clean_text = "".join([line for line in text.splitlines()
   #                         if "<|im_start|>" not in line and "<|im_end|>" not in line])
#    return clean_text.strip()

# Run and save to CSV
#with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
 #   writer = csv.writer(f)
  #  writer.writerow(["Prompt", "FP16_Output", "Q4_K_M_Output"])
    
   # for prompt in PROMPTS:
    #    print(f"Running FP16 model for prompt: {prompt}")
     #   fp16_out = run_model(FP16_MODEL, prompt)
      #  print(f"Running Q4_K_M model for prompt: {prompt}")
       # q4_out = run_model(Q4_MODEL, prompt)
        #writer.writerow([prompt, fp16_out, q4_out])

#print(f"All outputs saved in {OUTPUT_CSV}")







import subprocess
import time
import csv
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
MODELS = {
    "FP16": "models/llama-2-7b-fp16.gguf",
    "Q4_K_M": "models/llama-2-7b-q4_K_M.gguf"
}

LLAMA_BIN = "llama.cpp/build/bin/llama-run"  # llama.cpp binary
THREADS = 4
MAX_TOKENS = 256

PROMPTS = [
    "Explain artificial intelligence in simple words.",
    "Summarize Python programming language.",
    "Write a short story about a robot.",
    "Explain the importance of data privacy.",
    "How does blockchain work?",
    "Explain reinforcement learning in AI.",
    "Describe quantum computing.",
    "Write a Python function to reverse a string.",
    "Summarize the book 'To Kill a Mockingbird'.",
    "Explain the theory of relativity.",
    "Describe climate change and its effects.",
    "Write a short poem about the ocean.",
    "Explain the difference between supervised and unsupervised learning.",
    "Describe how a neural network works.",
    "Explain the concept of recursion in programming.",
    "Summarize the main points of the US Constitution.",
    "Write a short story about a magical forest.",
    "Explain the difference between AI and machine learning.",
    "Describe the process of photosynthesis.",
    "Explain the significance of the internet in modern society."
]

OUTPUT_CSV = Path("benchmark_tokens_sec.csv")

# -----------------------------
# RUN BENCHMARK
# -----------------------------
results = []

for model_name, model_path in MODELS.items():
    for prompt in PROMPTS:
        print(f"Running {model_name} on prompt: {prompt}")
        start_time = time.time()
        try:
            output = subprocess.run(
                [LLAMA_BIN, model_path, prompt, f"--threads={THREADS}", f"--max-tokens={MAX_TOKENS}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            ).stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running {model_name}: {e}")
            output = ""

        elapsed = time.time() - start_time
        # Remove llama.cpp tokens
        clean_output = output.replace("<|im_start|>assistant", "").replace("<|im_end|>", "").strip()
        tokens = len(clean_output.split())

        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
        print(f"Tokens: {tokens}, Time: {elapsed:.2f}s, Tokens/sec: {tokens_per_sec:.2f}\n")

        results.append({
            "Model": model_name,
            "Prompt": prompt,
            "Tokens": tokens,
            "Time_sec": round(elapsed, 2),
            "Tokens_per_sec": round(tokens_per_sec, 2)
        })

# -----------------------------
# SAVE TO CSV
# -----------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Model", "Prompt", "Tokens", "Time_sec", "Tokens_per_sec"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nBenchmark completed. Results saved to {OUTPUT_CSV}")
