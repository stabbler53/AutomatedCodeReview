from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from .code_analysis import analyze_code, evaluate_code_quality, provide_feedback
from .codemodel import model, tokenizer
from django.http import JsonResponse
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.cuda.amp import autocast


def index(request):
    return render(request, 'index.html')


def analyze_code_view(request):
    if request.method == 'POST':
        code = request.POST.get('code')
        findings = analyze_code(code)
        quality = evaluate_code_quality(findings)
        feedback = provide_feedback(findings)
        context = {
            'findings': findings,
            'quality': quality,
            'feedback': feedback
        }
        return render(request, 'service.html', context)
    return render(request, 'service.html')




def about(request):
  """ Renders the about template """
  # You can add logic here to retrieve data or perform actions
  # before rendering the template, if needed
  return render(request, 'about.html')

def chat(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-0.5B-Chat",
            torch_dtype="auto",
            device_map="auto"
        ).eval().to(device)
        model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

        # Prepare input
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

        # Generate response
        with torch.cuda.amp.autocast():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )

        # Post-process and decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        context = {
            'response': response,
            'prompt': prompt
        }
        return render(request, 'chat.html', context)

    return render(request, 'chat.html')


def project(request):
  """ Renders the project template """
  # You can add logic here to retrieve project data or display
  # project information, if needed
  return render(request, 'project.html')

def service(request):
    
    return render(request, 'service.html')

def team(request):
  """ Renders the team template """
  # You can add logic here to retrieve team member data or display
  # team information, if needed
  return render(request, 'team.html')

def not_found_404(request):
  """ Renders the 404 Not Found template """
  return render(request, '404.html')
