        
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from functions import get_completion

# Example usage
if __name__ == "__main__":
    messages = [
        SystemMessage(
            content="Be a helpful assistant and provide a response to the following message."
        ),
        HumanMessage(
            content="Hello, how are you? Please answer in one sentence maximum."
        )
    ]
    model = "gpt-4o"
    temperature = 0.7
    
    completion = get_completion(messages, model, temperature)
    print(completion)