from dotenv import load_dotenv
import os
from openai import OpenAI
import tiktoken

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Create the OpenAI client object
client = OpenAI(api_key=api_key)

class FluffEvaluator:
    BASE_INSTRUCTION = (
        "Your purpose is to critically evaluate a number of texts on their level of Fluff. "
        "Fluff refers to any content that is unnecessary, lacks substance, or does not directly contribute "
        "to the main message or purpose of a given text."
    )
    FLUFF_SCORES = "1 (no Fluff), 2 (Little Fluff), 3 (Some Fluff), 4 (Considerable Fluff), 5 (Too much Fluff)."

    def __init__(self, client):
        self.client = client

    def get_gpt_completion(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.0,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None

    @staticmethod
    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo-0613":
            num_tokens = 0
            for message in messages:
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += -1
            num_tokens += 2
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


class VanillaFluffEvaluator(FluffEvaluator):
    VANILLA_SCORE_INSTRUCTION = f"Provide a numerical Fluff score for this text according to the following scale: {FluffEvaluator.FLUFF_SCORES} Your only output is the score (1-5)."
    VANILLA_REASON_INSTRUCTION = "Provide a concise reason for your score in less than 15 words based on the text, your score and your understanding of Fluff."

    def evaluate(self, text):
        """
        Evaluate the fluff content of the given text and provide a score and reason.
        
        Args:
        text (str): The text to be evaluated.
        
        Returns:
        tuple: A tuple containing the score (int) and reason (str).
        """
        # First, get the fluff score
        messages = [
            {"role": "system", "content": self.BASE_INSTRUCTION + self.VANILLA_SCORE_INSTRUCTION},
            {"role": "user", "content": text}
        ]
        score_content = self.get_gpt_completion(messages)
        
        if score_content is None:
            return None, "Failed to get score."

        try:
            score = int(score_content.strip())
        except ValueError:
            return None, "Failed to parse score."

        # Second, get the reason for the fluff score
        messages = [
            {"role": "system", "content": self.BASE_INSTRUCTION + self.VANILLA_SCORE_INSTRUCTION},
            {"role": "user", "content": text},
            {"role": "assistant", "content": score_content},
            {"role": "user", "content": self.VANILLA_REASON_INSTRUCTION}
        ]
        reason = self.get_gpt_completion(messages)

        if reason is None:
            return score, "Failed to get reason."

        return score, reason.strip()


# # Example Usage
# vanilla_evaluator = VanillaFluffEvaluator(client)

# text_to_evaluate = "This is an example text to evaluate for fluff."

# vanilla_score, vanilla_reason = vanilla_evaluator.evaluate(text_to_evaluate)

# print(f"Vanilla Score: {vanilla_score}, Reason: {vanilla_reason}")
