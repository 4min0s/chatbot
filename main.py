from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google import genai


def traduction(user_text,result):
    if (result[0]["label"]!="en"):
        response = client.models.generate_content(
        model="gemini-2.5-flash", contents="return only the sentence nothing more translate the following text to English: " + user_text)
        return response.text
    return

#gemini setup 
client = genai.Client(api_key="AIzaSyC8l8Bv7nDRwFPy59KPDmb9TobSxYXYpqk")

# Load a pretrained language identification model based on XLM-RoBERTa
model_name = "papluca/xlm-roberta-base-language-detection"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for language detection
language_detector = pipeline("text-classification", model=model, tokenizer=tokenizer)

#  input
while(True):
    user_text =  input("Enter text to detect its language: ")

    #detection de la langue
    result = language_detector(user_text)

    print(result)

    #Taduction
    translated_text = traduction(user_text,result)
    print("Translated Text: ", translated_text)


