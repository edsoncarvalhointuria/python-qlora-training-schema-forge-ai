from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model = "gpt2"

brain = AutoModelForCausalLM.from_pretrained(model)
specialist_model = PeftModel.from_pretrained(brain, "./meu-modelo/checkpoint-25")
translate = AutoTokenizer.from_pretrained(model)
translate.pad_token = translate.eos_token
translate.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

question = [{"role": "user", "content": '{"nome": "LL"}'}]

input_ids = translate.apply_chat_template(question, return_tensors="pt")
saida_ids = specialist_model.generate(input_ids, max_new_tokens=50)
response = translate.decode(saida_ids[0])

print("resposta", response)
