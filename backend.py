

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


def TFIDF_query(query):
  query_vector = vectorizer.transform([preprocess_text(query)])

  cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

  # Get indices of the most similar products
  most_similar_indices = cosine_similarities.argsort()[::-1]

  # Keep track of recommended product IDs to ensure uniqueness
  recommended_product_ids = set()

  # Print the most similar unique products
  print("Top 20 recommended unique products:")
  productData = []
  for index in most_similar_indices:
      product_id = df.iloc[index]['ProductId']
      
      # Check if the product ID has already been recommended
      if product_id not in recommended_product_ids:
          recommended_product_ids.add(product_id)
          product_name = df.iloc[index]['productName']
          product_data_string = f"{product_name} (ID: {product_id})"
          productData.append(product_data_string)
        
      # Stop when you have enough unique recommendations
      if len(recommended_product_ids) == 20:
          break
  return recommended_product_ids , productData


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

def word2Vec(query, recommended_product_ids):
  # Read the CSV file into a DataFrame
  selected_rows = df[df['ProductId'].isin(recommended_product_ids)]


  text_lower_tokens = []
  for text in selected_rows['Text']:
    text_tokens = nltk.word_tokenize(text.lower())
    text_lower_tokens.append(text_tokens)

  model = gensim.models.Word2Vec(sentences=text_lower_tokens, vector_size=100, window=5, min_count=1, workers=1, epochs=20, seed=0)


  IN_embs = model.wv
  embeddings = IN_embs[query]
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


  productData = []

  for id in top5List:
      selected_row = df[df['ProductId'] == id]

      if not selected_row.empty:
          product_name = selected_row['productName'].iloc[0]
          summary = selected_row['Summary'].iloc[0]

          product_info_string = f"ProductId: {id}, ProductName: {product_name}, Summary: {summary}"

          productData.append(product_info_string)
      else:
          productData.append(f"ProductId {id} not found in the dataset.")
  print (productData)
  return productData


########################## order top 20 by rating ########################

# Filter the DataFrame to include only the specified ProductIds
def ScoreSort(recommended_product_ids):
    idList = list(recommended_product_ids)
    selected_rows = df[df['ProductId'].isin(idList)]

    average_scores = selected_rows.groupby('ProductId')['Score'].mean().reset_index()

    sorted_scores = average_scores.sort_values(by='Score', ascending=False)

    productData = []

    for _, row in sorted_scores.iterrows():
        product_id = row['ProductId']
        average_score = row['Score']
        #product_name = row['productName']

        # Format the information into a string
        product_info_string = f"ProductId: {product_id}, Average Score: {average_score}"

        print(product_info_string)

        productData.append(product_info_string)

    return productData


## recList = TFIDF_query("dog food")
#word2Vec("good", recList[0])
#ScoreSort(recList[0])
tflist = []



############################### FLASK API IMPLEMENTATION ##################################
from flask import Flask, request, jsonify

app = Flask(__name__)
### MUST CALL TFIDF FIRST
@app.route('/TFIDF', methods=['GET'])
def tf_idf_call():
    ## QUERY
    query = request.args.get('query', default='', type=str)


    global tflist
    result = TFIDF_query("query")
    tflist = result[0]
    output = {'results': list(result[1]), 'status': 'success'}

    return jsonify(output)

@app.route('/word2vec', methods=['GET'])
def word2vec_call():
    # QUERY

    result = word2Vec("yummy", tflist)
    output = {'results': result, 'status': 'success'}

    return jsonify(output)


@app.route('/ranking', methods=['GET'])
def rank_call():

    result = ScoreSort(tflist)
    output = {'results': list(result), 'status': 'success'}

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)


