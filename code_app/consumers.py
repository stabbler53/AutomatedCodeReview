# consumers.py
from channels.generic.websocket import WebsocketConsumer
import json
from .codemodel import model, tokenizer
import torch
from torch.cuda.amp import autocast

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        prompt = text_data_json['prompt']
        
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
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
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
        
        self.send(text_data=json.dumps({
            'message': response
        }))
