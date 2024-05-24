from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.cuda.amp import autocast

device = "cuda"  # the device to load the model onto

# Load model with gradient checkpointing enabled and optimized device mapping
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype="auto",
    device_map="auto"
).eval()
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

# Get user input
prompt = input("Please enter your prompt: ")

# Prepare the input
messages = [
    {"role": "system", "content": "You are a helpful assistant integrated into a site called CodeGuardian which helps users reviewing their code."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Use mixed precision for generation
with autocast():
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

# Post-process generated ids to remove the input prompt from the output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode the generated ids to text
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
