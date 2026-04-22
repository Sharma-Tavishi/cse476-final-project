import os


API_KEY  = os.getenv("OPENAI_API_KEY", "sk-mh4JzIDKRc4vcvmFonXWpA")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")


parser = argparse.ArgumentParser(description="Generate answers for the final project.")
parser.add_argument("--local", action="store_true", help="Run on local model.")

args = parser.parse_args()

global API_KEY
global API_BASE
global MODEL

if args.local:
    print("Using local model")
    API_KEY  = os.getenv("OPENAI_API_KEY", "lmstudio")
    API_BASE = os.getenv("API_BASE", "http://localhost:1234/v1")
    MODEL    = os.getenv("MODEL_NAME", "qwen/qwen3-30b-a3b-2507")

from openai import OpenAI

# Initialize the client with the ASU API endpoint
client = OpenAI(
    base_url=API_BASE,
    api_key=f"{API_KEY}",
)

response = client.chat.completions.create(
    model=f"{MODEL}",
    messages=[
        {"role": "user", "content": "Write a hello world program in Python."},
    ],
)

print(response.choices[0].message.content)