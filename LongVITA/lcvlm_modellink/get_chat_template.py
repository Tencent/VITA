



from transformers import AutoTokenizer

chat = [
      # {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
      {"role": "user", "content": "I'd like to show off how chat templating works!"},
    
]

tokenizer = AutoTokenizer.from_pretrained("./mistralai/Mistral-Nemo-Instruct-2407")
message = tokenizer.apply_chat_template(chat, tokenize=False)
print(message)


print("=" * 100)

tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2.5-14B-Instruct")
message = tokenizer.apply_chat_template(chat, tokenize=False)
print(message)
