from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

base_instruction = "Your purpose is to evaluate and score text on their fluff content on a scale from 1 (a lot of fluff) to 5 (no fluff). Your only output is the score. (1-5)"
fluff_definition = "Fluff is defined as content that is unnecessary, redundant, or overly wordy."
interest_categoriess = {
        'General Interest': 'Written to appeal to a broad audience, covering topics intended to engage or entertain readers without requiring specialized knowledge.',
        'Special Interest': 'Written to appeal to a pre-informed audience with personal interest, providing information required for decision making.',
        'Professional Interest': 'Written to appeal to a professional audience with deep prior knowledge, describing products and services for professional use.'
        }


def vanilla_fluff(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": base_instruction + fluff_definition},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content