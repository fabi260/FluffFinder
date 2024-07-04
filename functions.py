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
from openai import OpenAI


# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_KEY")
os.environ["anthropic_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Mapping of easy model names to actual model identifiers
MODEL_MAPPING = {
    "mistral7b": {"model": "mistralai/Mistral-7B-Instruct-v0.3", "provider": "deepinfra"},
    "mixtral8x22b": {"model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "provider": "deepinfra"},
    "llama370b": {"model": "meta-llama/Meta-Llama-3-70B-Instruct", "provider": "deepinfra"},
    "llama38b": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "provider": "deepinfra"},
    "llama38b_private": {"model": "gpt-3.5-turbo", "provider": "ommax_vllm"}, # model name is faked for openai compatibility
    "claude3opus": {"model": "claude-3-opus-20240229", "provider": "anthropic"},
    "claude3sonnet": {"model": "claude-3-sonnet-20240229", "provider": "anthropic"},
    "claude3haiku": {"model": "claude-3-haiku-20240307", "provider": "anthropic"},

    "qwen2": {"model": "Qwen/Qwen2-72B-Instruct", "provider": "deepinfra"},
    "phi3": {"model": "microsoft/Phi-3-medium-4k-instruct", "provider": "deepinfra"},
    "wizardLM28x22b": {"model": "microsoft/WizardLM-2-8x22B", "provider": "deepinfra"},
    "wizardLM27b": {"model": "microsoft/WizardLM-2-7B", "provider": "deepinfra"},
    "gemma": {"model": "google/gemma-1.1-7b-it", "provider": "deepinfra"},
    "WizardLM27b": {"model": "microsoft/WizardLM-2-7B", "provider": "deepinfra"},
    "nemotron4340b": {"model": "nvidia/Nemotron-4-340B-Instruct", "provider": "nvidia"},
}

# Define constants
BASE_INSTRUCTION = "Your purpose is to critically evaluate a number of texts on their level of Fluff. Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute to the main message or purpose of a given text."
FLUFF_SCORES = "1 (no Fluff), 2 (Little Fluff), 3 (Some Fluff), 4 (Considerable Fluff), 5 (Too much Fluff)."
VANILLA_SCORE_INSTRUCTION = f"Provide a numerical Fluff score for this text according to the following scale: {FLUFF_SCORES} Your ONLY output is one numerical number (1-5)."
VANILLA_REASON_INSTRUCTION = "Provide a concise reason for your score in less than 15 words based on the text, your score and your understanding of Fluff."


def nvidia_completion(messages, model, temperature):
    llm = ChatOpenAI(
        api_key= os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
        temperature=temperature,
        model=model,
    )
    try:
        response = llm.invoke(messages)
        return response.content
        # response = llm.chat.completions.create(
        # model=model, #"nvidia/Nemotron-4-340B-Instruct",
        # messages=messages,
        # temperature=temperature
        # )
        # return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

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
def ommax_completion(messages, model, temperature):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OMMAX_API_KEY"),
        base_url="http://www.delphi-dialogue.com:8000/v1"
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
    elif provider == "ommax_vllm":
        return ommax_completion(messages, actual_model, temperature)
    elif provider == "nvidia":
        return nvidia_completion(messages, actual_model, temperature)
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

def multi_agent_score(text, model, temperature, reason=False, print=False):
    
    # Agent 1 evaluation
    messages = [
        SystemMessage(
            content=BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION
        ),
        HumanMessage(
            content=text
        )
    ]
    score_content = get_completion(messages, model, temperature)

    # Agent 2 evaluation
    messages = [
        SystemMessage(
            content=BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION
        ),
        HumanMessage(
            content=text
        )
    ]
    score_content = get_completion(messages, model, temperature)

    # Moderator agent summary
    messages = [
        SystemMessage(
            content=BASE_INSTRUCTION + VANILLA_SCORE_INSTRUCTION
        ),
        HumanMessage(
            content=text
        )
    ]
    score_content = get_completion(messages, model, temperature)




# Kippendorff's alpha analysis
def kippendorff_analysis(value_counts, level_of_measurement='interval', out='data'):
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