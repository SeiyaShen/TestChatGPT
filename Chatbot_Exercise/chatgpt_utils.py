# embedding_utils.py

import openai
import os
import re
import requests
import sys
from num2words import num2words
import os
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

# search through the reviews for a specific product
def search_docs( user_query, top_n = 3, to_print = False):
    url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

    r = requests.get(url, headers={"api-key": API_KEY})

    # This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
    df=pd.read_csv(os.path.join(os.getcwd(),r'C:\Users\seiya.z.shen\OneDrive - Avanade\Data_AI\Open_AI\bill_sum_data.csv')) 

    # DataFrame called df_bills which will contain only the columns for text, summary, and title.
    df_bills = df[['text', 'summary', 'title']]

    #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters
    pd.options.mode.chained_assignment = None 

    df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

    # need to remove any bills that are too long for the token limit (8192 tokens).
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
    df_bills = df_bills[df_bills.n_tokens < 8192]

    # To understand the n_tokens column a little more as well how text ultimately is tokenized, 
    sample_encode = tokenizer.encode(df_bills.text[0]) 
    decode = tokenizer.decode_tokens_bytes(sample_encode)

    # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    df_bills['ada_v2'] = df_bills["text"].apply(lambda x : get_embedding(x, engine = 'Seiya-embedding-ada-002' ))

    embedding = get_embedding(
        user_query,
        engine = 'Seiya-embedding-ada-002' # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df_bills["similarities"] = df_bills.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df_bills.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if res.empty:
        return "Sorry, I cannot find the answer"
    
    if to_print:
        print(res) # original is display(res)
    return res["summary"].iloc[0]

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