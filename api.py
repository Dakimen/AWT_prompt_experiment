from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

client = OpenAI(timeout=30.0, max_retries=1)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Describe one visible characteristic of a wooden chair in one sentence."
        }
    ],
    temperature=0.5,
    max_tokens=80
)

print("Model:", response.model)
print("Response:", response.choices[0].message.content)
print("Usage:", response.usage)
