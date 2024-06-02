import logging
import os
import torch
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from .code_analysis import analyze_code, evaluate_code_quality, provide_feedback
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s:%(message)s')

# Load model and tokenizer outside of the view function
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype="auto",
    device_map="auto"
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

def handle_uploaded_file(uploaded_file):
    file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
    file_full_path = os.path.join(default_storage.location, file_path)
    try:
        with open(file_full_path, 'r') as file:
            code = file.read()
    except Exception as e:
        logging.error(f"Error reading uploaded file: {e}")
        raise
    finally:
        try:
            default_storage.delete(file_path)
        except Exception as e:
            logging.error(f"Error deleting file: {e}")
    return code

def get_model_response(prompt):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant integrated into a site called CodeGuardian which helps users reviewing their code."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)  # Move input tensors to the same device as the model
        with torch.cuda.amp.autocast():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        logging.error(f"Error generating model response: {e}")
        return "An error occurred while generating the response."

def index(request):
    return render(request, 'index.html')

def analyze_code_view(request):
    findings = []
    feedback = []
    quality = {
        'total_functions': 0,
        'total_issues': 0,
        'quality_score': 0,
    }
    
    if request.method == 'POST':
        code = request.POST.get('code', '').strip()
        uploaded_file = request.FILES.get('file')
        
        if uploaded_file:
            try:
                code = handle_uploaded_file(uploaded_file)
            except Exception:
                return render(request, 'service.html', {'error': 'Error reading uploaded file.'})
        
        if code:
            try:
                findings = analyze_code(code)
                quality = evaluate_code_quality(findings)
                feedback = provide_feedback(findings)
            except Exception as e:
                logging.error(f"Error analyzing code: {e}")
                return render(request, 'service.html', {'error': 'Error analyzing code.'})
        else:
            return render(request, 'service.html', {'error': 'Please provide code in the textarea or upload a file.'})
    
    context = {
        'findings': findings,
        'quality': quality,
        'feedback': feedback
    }
    
    return render(request, 'service.html', context)

def about(request):
    return render(request, 'about.html')

def chat(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        response = get_model_response(prompt)
        context = {
            'response': response,
            'prompt': prompt
        }
        return render(request, 'chat.html', context)
    
    return render(request, 'chat.html')

def service(request):
    return render(request, 'service.html')

def team(request):
    return render(request, 'team.html')
