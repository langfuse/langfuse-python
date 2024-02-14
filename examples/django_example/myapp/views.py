from django.shortcuts import render
from django.http import JsonResponse, HttpResponseServerError
from myapp.langfuse_integration import get_response_openai, langfuse

def main_route(request):
    return JsonResponse({"message": "Hey, this is an example showing how to use Langfuse with Django."})

def campaign(request):
    prompt = request.GET.get('prompt', '')
    response = get_response_openai(prompt)
    return JsonResponse(response)
