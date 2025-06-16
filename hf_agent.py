from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, 
                   max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)

while True:
    user_input = input("\nAsk me anything (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input, truncation=True)
    print("\nAI Agent's Response:")
    print(response[0]["generated_text"])
