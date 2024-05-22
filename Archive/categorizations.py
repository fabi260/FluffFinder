from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


def perspective(text):
    perspective_instruction = "Your purpose is to identify the perspective of the text. Your only output is either 'Partial' or 'Impartial'."
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", temperature=0.0, # Temperature of 0.0 ensures deterministic results
        messages=[
            {"role": "system", "content": perspective_instruction},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content

def perspective(text):
    perspective_instruction = "Your purpose is to identify the perspective of the text. Your only output is either 'Partial' or 'Impartial'."
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", temperature=0.0, # Temperature of 0.0 ensures deterministic results
        messages=[
            {"role": "system", "content": perspective_instruction},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content