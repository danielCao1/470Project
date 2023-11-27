
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


####################### TF-IDF SIMILAR ITEMS #############################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Read the CSV file into a DataFrame
file_path = 'amazonFood.csv'
df = pd.read_csv(file_path)

# Assume 'productName' contains information about the products
product_names = df['productName'].tolist()

# Tokenize and preprocess the product names
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

preprocessed_product_names = [preprocess_text(name) for name in product_names]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_product_names)

# Calculate cosine similarity between the query and each product
query = "barbeque chips"
query_vector = vectorizer.transform([preprocess_text(query)])

cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Get indices of the most similar products
most_similar_indices = cosine_similarities.argsort()[::-1]

# Keep track of recommended product IDs to ensure uniqueness
recommended_product_ids = set()

# Print the most similar unique products
print("Top 20 recommended unique products:")
for index in most_similar_indices:
    product_id = df.iloc[index]['ProductId']
    
    # Check if the product ID has already been recommended
    if product_id not in recommended_product_ids:
        recommended_product_ids.add(product_id)
        print(df.iloc[index]['productName'])
    
    # Stop when you have enough unique recommendations
    if len(recommended_product_ids) == 20:
        break
print (recommended_product_ids)


###################### USE WORD2VEC ON TEXT #######################


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

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

selected_rows = df[df['ProductId'].isin(recommended_product_ids)]


text_lower_tokens = []
for text in selected_rows['Text']:
  text_tokens = nltk.word_tokenize(text.lower())
  text_lower_tokens.append(text_tokens)

model = gensim.models.Word2Vec(sentences=text_lower_tokens, vector_size=100, window=5, min_count=1, workers=1, epochs=20, seed=0)


IN_embs = model.wv
embeddings = IN_embs["delicious"]
scores = []

for i, doc in enumerate(recommended_product_ids):
  docAry = []
  for token in text_lower_tokens[i]:
    docAry.append(IN_embs[token])
  embDocAry = np.mean(docAry, axis = 0)
  cos = 1 - cosine(embeddings, embDocAry)
  scores.append((i, cos))

ranked_score = sorted(scores , key=lambda x:x[1], reverse = True)
idList = list(recommended_product_ids)
top5List = []
for i in range(5):
  top5List.append(idList[ranked_score[i][0]])
  print (ranked_score[i][1] , '|', idList[ranked_score[i][0]] )



for id in top5List:
  selected_row = df[df['ProductId'] == id]

  # Check if the ProductId is present in the DataFrame
  if not selected_row.empty:
      # Print the corresponding productName and Summary
      product_name = selected_row['productName'].iloc[0]
      summary = selected_row['Summary'].iloc[0]

      print(f"ProductId: {id}")
      print(f"ProductName: {product_name}")
      print(f"Summary: {summary}")
  else:
      print(f"ProductId {id} not found in the dataset.")


########################## order top 20 by rating ########################

# Filter the DataFrame to include only the specified ProductIds
selected_rows = df[df['ProductId'].isin(idList)]

# Calculate the average score for each product in the selected rows
average_scores = selected_rows.groupby('ProductId')['Score'].mean().reset_index()

# Print the average scores
print("Average Scores:")
for _, row in average_scores.iterrows():
    print(f"ProductId: {row['ProductId']}, Average Score: {row['Score']}")