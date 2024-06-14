from dotenv import load_dotenv
import os
import tiktoken
from krippendorff import alpha
from bootstrap_alpha import bootstrap
import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatDeepInfra
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_KEY")
os.environ["anthropic_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Mapping of easy model names to actual model identifiers
MODEL_MAPPING = {
    "mistral7b": {"model": "mistralai/Mistral-7B-Instruct-v0.3", "provider": "deepinfra"},
    "mixtral8x22b": {"model": "mistralai/Mistral-7B-Instruct-v0.3", "provider": "deepinfra"},
    "llama370b": {"model": "meta-llama/Meta-Llama-3-70B-Instruct", "provider": "deepinfra"},
    "llama38b": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "provider": "deepinfra"},
    "claude3opus": {"model": "claude-3-opus-20240229", "provider": "anthropic"},
    "claude3sonnet": {"model": "claude-3-sonnet-20240229", "provider": "anthropic"},
    "claude3haiku": {"model": "claude-3-haiku-20240307", "provider": "anthropic"}
}

# Define constants
BASE_INSTRUCTION = "Your purpose is to critically evaluate a number of texts on their level of Fluff. Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute to the main message or purpose of a given text."
FLUFF_SCORES = "1 (no Fluff), 2 (Little Fluff), 3 (Some Fluff), 4 (Considerable Fluff), 5 (Too much Fluff)."
VANILLA_SCORE_INSTRUCTION = f"Provide a numerical Fluff score for this text according to the following scale: {FLUFF_SCORES} Your ONLY output is one numerical number (1-5)."
VANILLA_REASON_INSTRUCTION = "Provide a concise reason for your score in less than 15 words based on the text, your score and your understanding of Fluff."

# Define wrapper functions for AI model completions
def openai_completion(messages, model, temperature):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error: {e}")
        return None
# Deepinfra completion
def deepinfra_completion(model, prompt, temperature):
    llm = ChatDeepInfra(model_id=model)
    llm.model_kwargs = {
        "temperature": temperature
    }
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Anthropic completion
def anthropic_completion(model, prompt, temperature):
    llm = ChatAnthropic(model=model, temperature=temperature)
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Main function to get completion from the specified model
def get_completion(messages, model, temperature, api_key=None):
    # Use model mapping to get the actual model identifier and provider
    if model.startswith("gpt-"):
        actual_model = model
        provider = "openai"
    else:
        model_info = MODEL_MAPPING.get(model)
        if not model_info:
            raise ValueError(f"Unsupported model: {model}")
        
        actual_model = model_info["model"]
        provider = model_info["provider"]

    if provider == "openai":
        return openai_completion(messages, actual_model, temperature)
    elif provider == "deepinfra":
        return deepinfra_completion(actual_model, messages, temperature)
    elif provider == "anthropic":
        return anthropic_completion(actual_model, messages, temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Vanilla Fluff Evaluator
def vanilla_score(text, model, temperature, reason=False):
    # Get the fluff score
    messages = [
        SystemMessage(
            content=BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION
        ),
        HumanMessage(
            content=text
        )
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
        messages.append(AIMessage(
            content=score_content
        ))
        messages.append(HumanMessage(
            content=VANILLA_REASON_INSTRUCTION
        ))
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