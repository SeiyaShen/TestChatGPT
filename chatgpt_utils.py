# embedding_utils.py

import openai
import re
import requests
import sys
from num2words import num2words
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken



# API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
# RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 

#my new end point
API_KEY =  "cbc8848f798b4770bb144e69bfc5757a"
RESOURCE_ENDPOINT = "https://openai-studio-seiya.openai.azure.com/"

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
#openai.api_version = "2022-12-01"
openai.api_version = "2023-03-15-preview" 



def get_answer( question ):
    model_engine = "gpt-35-turbo"

    # response = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=question,
    #     temperature=0.3,
    #     max_tokens=1024,
    #     stop=None)

    conversation=[{"role": "system", "content": "You are a helpful assistant."}]

    conversation.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        engine = model_engine, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages = conversation
    )
    
    #Response = response.choices[0].text.strip()#old format

    Response = response['choices'][0]['message']['content']

    # print(response)
    # print("hello seiya demo")
    # print("-------------------")
    # print(Response)
    return Response


#
# def get_embedding(text, engine_in):
#     text = text.replace("\n", " ")
#     return openai.Embedding.create(input=[text], engine=engine_in)["data"][0]["embedding"]

# def cosine_similarity(a, b):
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     return dot_product / (norm_a * norm_b)

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s
