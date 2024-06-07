from dotenv import load_dotenv
import os
from openai import OpenAI
import tiktoken
from krippendorff import alpha
from bootstrap_alpha import bootstrap
import numpy as np

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Create the OpenAI client object
client = OpenAI(api_key=api_key)

# Define constants
BASE_INSTRUCTION = "Your purpose is to critically evaluate a number of texts on their level of Fluff. Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute to the main message or purpose of a given text."
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
def vanilla_score(text, model, temperature, reason=False):
    # Get the fluff score
    messages = [
        {"role": "system", "content": BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION},
        {"role": "user", "content": text}
    ]
    score_content = get_completion(messages, model, temperature)
    
    if score_content is None:
        return "Failed to get score."
    try:
        score = int(score_content.strip())
    except ValueError:
        return "Failed to parse score."
    
    # Get the reason for the fluff score if requested
    if not reason:
        return score
    else:
        messages.append({"role": "assistant", "content": score_content})
        messages.append({"role": "user", "content": VANILLA_REASON_INSTRUCTION})
        reason = get_completion(messages, model, temperature)
        if reason is None:
            return score, "Failed to get reason."
        return score, reason.strip()

# Kippendorff's alpha analysis
def kippendorff_analysis(value_counts, level_of_measurement='ordinal', out='data'):
    # Calculate Krippendorff's alpha
    value_domain=value_counts.columns.values.astype(int)

    # check that at least one text was evaluated by multiple people
    if value_counts.sum(axis=1).mean() <= 1 or value_counts.std().mean() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        k_alpha = alpha(value_counts=value_counts, level_of_measurement=level_of_measurement, value_domain=value_domain)

        ci_95_2s, est = bootstrap(value_counts=value_counts, level_of_measurement=level_of_measurement, num_iterations=1000, confidence_level=0.95, sampling_method= 'krippendorff', return_bootstrap_estimates=True, value_domain=value_domain)
        
        lb_ci_95_1s = np.percentile(est, 5)
        confidence_at_least_667 = np.mean(est >= 0.667)
        confidence_at_least_8 = np.mean(est >= 0.8)
        confidence_lessthan_667 = np.mean(est < 0.667)
        
        if out == 'text':
            return print(
            f"Krippendorff's alpha: {k_alpha:.3f}\n"
            f"95% CI: {ci_95_2s[0]:.3f} - {ci_95_2s[1]:.3f}\n"
            f"Confidence of data being reliable and alpha being at least 0.8: {confidence_at_least_8 * 100:.1f}%\n"
            f"Confidence of data being tentatively reliable and alpha being at least 0.667: {confidence_at_least_667 * 100:.1f}%\n"
            f"Confidence of data being unreliable and alpha being less than 0.667: {confidence_lessthan_667 * 100:.1f}%\n"
            f"Lower bound of 95% one-sided CI: {lb_ci_95_1s:.3f}"
        )
        elif out == 'data':
            return k_alpha, ci_95_2s, confidence_at_least_8, confidence_at_least_667, confidence_lessthan_667, lb_ci_95_1s
        else:
            raise ValueError(f"Unsupported output format: {out}")