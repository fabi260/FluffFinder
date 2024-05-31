from dotenv import load_dotenv
import os
from openai import OpenAI
import tiktoken

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Create the OpenAI client object
client = OpenAI(api_key=api_key)

# Define constants
BASE_INSTRUCTION = (
    "Your purpose is to critically evaluate a number of texts on their level of Fluff. "
    "Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute "
    "to the main message or purpose of a given text."
)
FLUFF_SCORES = "1 (no Fluff), 2 (Little Fluff), 3 (Some Fluff), 4 (Considerable Fluff), 5 (Too much Fluff)."
VANILLA_SCORE_INSTRUCTION = f"Provide a numerical Fluff score for this text according to the following scale: {FLUFF_SCORES} Your ONLY output is one numerical number (1-5)."
VANILLA_REASON_INSTRUCTION = "Provide a concise reason for your score in less than 15 words based on the text, your score and your understanding of Fluff."

# Define wrapper functions for AI model completions
def openai_completion(messages, model, temperature):
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to get completion from the specified model
def get_completion(messages, model, temperature): ##TODO: Add llama and claude models and properly integrate their keys
    prompt = "\n".join([msg["content"] for msg in messages]) # Combine messages into a single prompt for llama and claude models
    
    if model.startswith("gpt-"):
        return openai_completion(messages, model, temperature)
    # elif model == "llama":
    #     return llama_completion(prompt, api_key, temperature)
    # elif model == "claude":
    #     return claude_completion(prompt, api_key, temperature)
    else:
        raise ValueError(f"Unsupported model: {model}")

# Vanilla Fluff Evaluator
def vanilla_score(text, model, temperature):
    # Get the fluff score
    messages = [
        {"role": "system", "content": BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION},
        {"role": "user", "content": text}
    ]
    score_content = get_completion(messages, model, temperature)
    
    if score_content is None:
        return None, "Failed to get score."
    try:
        score = int(score_content.strip())
    except ValueError:
        return None, "Failed to parse score."
    
    # Get the reason for the fluff score
    messages = [
        {"role": "system", "content": BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION},
        {"role": "user", "content": text},
        {"role": "assistant", "content": score_content},
        {"role": "user", "content": VANILLA_REASON_INSTRUCTION}
    ]
    reason = get_completion(messages, model, temperature)

    if reason is None:
        return score, "Failed to get reason."
    
    return score, reason.strip()
