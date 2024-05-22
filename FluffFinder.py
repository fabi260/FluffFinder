from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Define the base instruction and related strings
BASE_INSTRUCTION = (
    "Your purpose is to critically evaluate a number of texts on their level of Fluff. "
    "Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute "
    "to the main message or purpose of a given text."
)
FLUFF_SCORES = "1 (no Fluff), 2 (Little Fluff), 3 (Some Fluff), 4 (Considerable Fluff), 5 (Too much Fluff)."
VANILLA_SCORE_INSTRUCTION = f"Provide a numerical Fluff score for this text according to the following scale: {FLUFF_SCORES} Your only output is the score (1-5)."
VANILLA_REASON_INSTRUCTION = "Provide a concise reason for your score in less than 15 words based on the text, your score and your understanding of Fluff."

def get_gpt_completion(messages):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def vanilla_fluff(text): ##TODO: Add possibility to only get score
    """
    Evaluate the fluff content of the given text and provide a score and reason.
    
    Args:
    text (str): The text to be evaluated.
    
    Returns:
    tuple: A tuple containing the score (int) and reason (str).
    """
    # First, get the fluff score
    messages = [
        {"role": "system", "content": BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION},
        {"role": "user", "content": text}
    ]
    score_content = get_gpt_completion(messages)
    
    if score_content is None:
        return None, "Failed to get score."

    try:
        score = int(score_content.strip())
    except ValueError:
        return None, "Failed to parse score."


    # Second, get the reason for the fluff score
    messages = [
        {"role": "system", "content": BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION},
        {"role": "user", "content": text},
        {"role": "assistant", "content": score_content},
        {"role": "user", "content": VANILLA_REASON_INSTRUCTION}
    ]
    reason = get_gpt_completion(messages)

    if reason is None:
        return score, "Failed to get reason."

    return score, reason.strip()