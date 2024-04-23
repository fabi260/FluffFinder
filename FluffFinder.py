from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

base_instruction = "Your role is to evaluate texts on their Fluff content. Fluff is defined as content that is unnecessary, redundant, or overly wordy. Your purpose is to score the text on its fluff content on a scale from 1 (a lot of fluff) to 5 (no fluff). Your only output is the score. (1-5)"


def vanilla_fluff(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content
