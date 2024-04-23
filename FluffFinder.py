from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",

  messages=[
    {"role": "system", "content": "Hi"},
    {"role": "user", "content": "Say: Hello world!"}
  ]
)

print(completion.choices[0].message)