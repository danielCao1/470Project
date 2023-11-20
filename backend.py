
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
texts_array = df['Text'].head(10000).tolist()
id_array = df['ProductId'].head(10000).tolist()
# Print the first few texts for verification
print(texts_array[:5])
print(id_array[:5])
count = 0


texts_lower_tokens = []
for text in texts_array:
  text_tokens  = nltk.word_tokenize(text.lower())
  texts_lower_tokens.append(text_tokens)

model = gensim.models.Word2Vec(sentences=texts_lower_tokens, vector_size=50, window=5, min_count=1, workers=1, epochs=20, seed=0)

IN_embs = model.wv
OUT_embs = model.syn1neg


embeddings = IN_embs["sweet"]
scores = []

for i, doc in enumerate(id_array):
  docAry = []
  for token in texts_lower_tokens[i]:
    docAry.append(IN_embs[token])
  embDocAry = np.mean(docAry, axis = 0)
  cos = 1 - cosine(embeddings, embDocAry)
  scores.append((i, cos))

ranked_score = sorted(scores , key=lambda x:x[1], reverse = True)
for i in range(10):
  print (ranked_score[i][1] , '|', id_array[ranked_score[i][0]] )

import requests
from bs4 import BeautifulSoup

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
