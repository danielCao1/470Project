
############## REACT API ##########################
# from flask import Flask, jsonify

# app = Flask('backend.py')

# @app.route('/api/data', methods=['GET'])
# def get_data():
#     data = {'message': 'Hello from Python!'}
#     return jsonify(data)

# if 'backend.py' == 'main':
#     app.run(debug=True)







# ############## CHATGPT API IMPLEMENTATION ################

# import openai

# # Set your OpenAI API key here
# api_key = "your-api-key"

# # Set the API endpoint
# api_endpoint = "https://api.openai.com/v1/chat/completions"

# # Set the conversation history
# conversation_history = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
# ]

# # Set the user input
# user_input = "It was played in Arlington, Texas."

# # Create the request payload
# payload = {
#     "model": "gpt-3.5-turbo",
#     "messages": [{"role": role, "content": content} for {"role", "content"} in conversation_history],
#     "max_tokens": 100,
#     "stop": None,
#     "temperature": 0.7,
# }

# # Add the user input to the payload
# payload["messages"].append({"role": "user", "content": user_input})

# # Make the API request
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}",
# }

# response = requests.post(api_endpoint, json=payload, headers=headers)

# # Print the API response
# print(response.json())




###################### Recommender Implementation #####################
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import random
from scipy.spatial.distance import cosine
import copy

import gensim
import json
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.tokenize.word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine



# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'amazonFood.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract the 'Text' column and fill an array with the texts
texts_array = df['Text'].head(40000).tolist()
id_array = df['ProductId'].head(40000).tolist()
name_array = df['productName'].head(40000).unique().tolist()
# Print the first few texts for verification
print(texts_array[:5])
print(id_array[:5])
count = 0

unique_id = df['ProductId'].head(40000).unique().tolist()
texts_lower_tokens = []

#a doc consists of all reviews for a certain product
for id in unique_id:
  rows = df.loc[df["ProductId"] == id]
  review = ""
  #print("rows",rows["ProductId"])
  for row in rows["Text"]:
    review += row
  text_tokens  = nltk.word_tokenize(review.lower())
  texts_lower_tokens.append(text_tokens)
  
print("size of unique id", len(unique_id))
print("size of texts_lower_tokens",len(texts_lower_tokens))

model = gensim.models.Word2Vec(sentences=texts_lower_tokens, vector_size=100, window=5, min_count=1, workers=1, epochs=20, seed=0)

IN_embs = model.wv
OUT_embs = model.syn1neg

queries = ["sweet","candy"]
embeddings = IN_embs["sweet"]
scores = [0]*len(texts_lower_tokens)

#multi-word queries
doc_embs = np.zeros(100)
for query in queries:
  queryEmb = IN_embs[query]
  for i, doc in enumerate(unique_id):
    docAry = np.zeros(100)
    for token in texts_lower_tokens[i]:
      #print(token)
      docAry += OUT_embs[model.wv.key_to_index[token]]
    embDocAry = docAry/len(texts_lower_tokens[i])
    cos = 1-  cosine(queryEmb, embDocAry)#1 - cosine(embeddings, embDocAry)
    scores[i] += cos

#average results for each queryword
scores = [x/len(queries) for x in scores]

#find indexes of highest scores
idx = list(np.argsort(scores))
idx.reverse()

#print highest scores
for i in idx[:25]:
  print (scores[i] , '|', unique_id[i])

import requests
from bs4 import BeautifulSoup

"""
# Replace '{productid}' with the actual product ID in the URL
for i in range(100):
    product_url = 'https://www.amazon.com/dp/{productid}'
    product_id = id_array[ranked_score[i][0]]

    # Make a request to the Amazon product page
    response = requests.get(product_url.format(productid=product_id))

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract information from the page as needed
        # Example: Print the product title
        product_title = soup.find('span', {'id': 'productTitle'}).text.strip()
        print('Product Title:', product_title)
    else:
        print('Failed to retrieve the page. Status code:', response.status_code)
"""