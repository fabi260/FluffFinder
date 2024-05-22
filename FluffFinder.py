from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# base_instruction = "Your purpose is to evaluate and score text on their fluff content on a scale from 1 (a lot of fluff) to 5 (no fluff). Your only output is the score. (1-5)"
base_instruction = "Your purpose is to critically evaluate a number of texts on their level of Fluff"
fluff_definition = "Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute to the main message or purpose of a given text."
fluff_scores = "1 (no Fluff), 2 (Little Fluff), 3 (Some Fluff), 4 (Considerable Fluff), 5 (Too much Fluff)."
output_format = "Numerical score (1-5), reason"
vanilla_instruction = f'Provide a numerical Fluff score for this text according to the following scale: {fluff_scores} Also provide a one-sentence reason for your score. Your output should have the following format: {output_format}'
# fluff_characteristics = 
interest_categoriess = {
        'General Interest': 'Written to appeal to a broad audience, covering topics intended to engage or entertain readers without requiring specialized knowledge.',
        'Special Interest': 'Written to appeal to a pre-informed audience with personal interest, providing information required for decision making.',
        'Professional Interest': 'Written to appeal to a professional audience with deep prior knowledge, describing products and services for professional use.'
        }


def vanilla_fluff(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", temperature=0.0, # Temperature of 0.0 ensures deterministic results
        messages=[
            {"role": "system", "content": base_instruction + fluff_definition + vanilla_instruction},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content.astype(int)