import streamlit as st
import requests
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# Configure Google Generative AI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Function to call Llama API
def call_llama(prompt):
    url = "http://34.132.107.175:11434/api/generate"
    payload = {
        "model": "llama3.1:latest",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Function to call DeepSeek API
def call_deepseek(prompt):
    url = "http://34.132.107.175:11434/api/generate"
    payload = {
        "model": "deepseek-v2:latest",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Function to call Azure OpenAI
def call_azure(prompt):
    azure_chat_llm = AzureChatOpenAI(
        deployment_name="gpt-4",
        azure_endpoint="https://skille-north.openai.azure.com/",
        openai_api_key="75ccb32e76c64317b5f0eeae888cc373",
        openai_api_version="2023-06-01-preview"
    )
    return azure_chat_llm(prompt)

# Function to call Google Gemini
def call_gemini(prompt):
    gemini = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5, api_key=GOOGLE_API_KEY)
    return gemini.invoke(prompt)

# Streamlit App
st.title("Ken42 LLM")
st.markdown("Select a model and run your query:")

# Model Selection
models = ["Llama 3.1", "DeepSeek v2", "Azure OpenAI GPT-4", "Google Gemini 1.5 Flash"]
selected_model = st.selectbox("Choose a model", models)

# Input for prompt
prompt = st.text_area("Enter your prompt", "")

# Submit Button
if st.button("Run"):
    if prompt.strip() == "":
        st.error("Please enter a prompt.")
    else:
        if selected_model == "Llama 3.1":
            result = call_llama(prompt)
        elif selected_model == "DeepSeek v2":
            result = call_deepseek(prompt)
        elif selected_model == "Azure OpenAI GPT-4":
            result = call_azure(prompt)
        elif selected_model == "Google Gemini 1.5 Flash":
            result = call_gemini(prompt)
        else:
            result = {"error": "Invalid model selected."}

        # Display the result
        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif "response" in result:
            st.write(result['response'])
        elif "content" in result:
            st.write(result['content'])
        else:
            st.write(result.content)
