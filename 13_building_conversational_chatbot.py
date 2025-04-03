from llama_cpp import Llama

from llama_cpp import Llama

class Conversation:
    def __init__(self, llm: Llama, system_prompt='', history=[]):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = [
            {'role': 'system', 'content': self.system_prompt}] + history

    def create_completion(self, user_prompt=''):
        self.history.append({'role': 'user', 'content': user_prompt})
        output = self.llm.create_chat_completion(messages=self.history)
        conversation_response = output['choices'][0]['message']
        self.history.append(conversation_response)
        return conversation_response['content']


def start_chat(llm: Llama, chatbot: Conversation):
    while True:
        user_input = input("Enter your text: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.create_completion(user_input)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-IQ3_M.gguf"
    )
    chatbot = Conversation(
        llm, system_prompt="You are a helpful Maths assistant. If you are asked outside maths, say 'I am a maths assistant, I can only help you with maths.'")
    start_chat(llm, chatbot)
