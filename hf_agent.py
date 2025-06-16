# Updated to DialoGPT version
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "microsoft/DialoGPT-medium"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create chatbot pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer,
                   max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# Chat loop
while True:
    user_input = input("\nAsk me anything (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input, truncation=True)
    print("\nAI Agent's Response:")
    print(response[0]["generated_text"])
