
import numpy as np
import pandas as pd
from collections import Counter
import math

# Dataset
documents = [
    "people watch campusx",
    "campusx watch campusx",
    "people write comment",
    "campusx write comment"
]


# Step 0: Preprocessing - Creating vocabulary
def create_vocabulary(documents):
    vocabulary = set()
    for doc in documents:
        for word in doc.split():
            vocabulary.add(word)
    return sorted(list(vocabulary))

vocabulary = create_vocabulary(documents)
print("Vocabulary:", vocabulary)


# Step 1: Term Frequency (TF)
def calculate_tf(document, vocabulary):
    words = document.split() #document.split()  # ['campusx', 'watch', 'campusx'] return list of strings (list[str])
    word_count = Counter(words)
    total_words = len(words)
    
    tf_dict = {}
    for term in vocabulary:
        tf_dict[term] = word_count[term] / total_words
    
    return tf_dict


# Calculate TF for each document
tf_values = []
for i, doc in enumerate(documents):
    tf_dict = calculate_tf(doc, vocabulary)
    tf_values.append(tf_dict)
    print(f"TF for Document {i+1}:", tf_dict)


# Step 2: Inverse Document Frequency (IDF)
def calculate_idf(documents, vocabulary):
    total_docs = len(documents)
    idf_dict = {}
    
    for term in vocabulary:
        doc_count = sum(1 for doc in documents if term in doc.split())
        idf_dict[term] = math.log(total_docs / doc_count)
    
    return idf_dict

idf_values = calculate_idf(documents, vocabulary)
print("\nIDF Values:", idf_values)


# Step 3: TF-IDF Calculation
def calculate_tfidf(tf_values, idf_values):
    tfidf_docs = []
    
    for tf_dict in tf_values:
        tfidf_dict = {}
        for term, tf in tf_dict.items():
            tfidf_dict[term] = tf * idf_values[term]
        tfidf_docs.append(tfidf_dict)
    
    return tfidf_docs

tfidf_values = calculate_tfidf(tf_values, idf_values)


# Display TF-IDF for each document
for i, tfidf_dict in enumerate(tfidf_values):
    print(f"\nTF-IDF for Document {i+1}:", tfidf_dict)


# Create TF-IDF Matrix
def create_tfidf_matrix(tfidf_values, vocabulary):
    matrix = []
    for tfidf_dict in tfidf_values:
        vector = [tfidf_dict[term] for term in vocabulary]
        matrix.append(vector)
    return matrix

tfidf_matrix = create_tfidf_matrix(tfidf_values, vocabulary)


# Create a DataFrame for better visualization
df_tfidf = pd.DataFrame(tfidf_matrix, columns=vocabulary)
df_tfidf.index = pd.Index([f"D{i+1}" for i in range(len(documents))])
print("\nTF-IDF Matrix:")
print(df_tfidf)


# Document Vector Representation
print("\nDocument Vectors:")
for i, vector in enumerate(tfidf_matrix):
    print(f"D{i+1}: {vector}")


# Additional: Document Similarity using Cosine Similarity
def custom_cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = math.sqrt(sum(a * a for a in vec1))
    norm_vec2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    
    return dot_product / (norm_vec1 * norm_vec2)


# Calculate similarities between documents
print("\nDocument Similarities (Cosine):")
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        similarity = custom_cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
        print(f"Similarity between D{i+1} and D{j+1}: {similarity:.4f}")



# *********************************************************************

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset
documents = [
    "people watch campusx",
    "campusx watch campusx",
    "people write comment",
    "campusx write comment"
]

# Step 1: Using TfidfVectorizer from sklearn to compute TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Display TF-IDF matrix
df_tfidf = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(df_tfidf)

# Document Vector Representation
print("\nDocument Vectors:")
for i, vector in enumerate(tfidf_matrix.todense()):
    print(f"D{i+1}: {vector}")

# Step 2: Document Similarity using Cosine Similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Display Document Similarities
print("\nDocument Similarities (Cosine):")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        similarity = similarity_matrix[i][j]
        print(f"Similarity between D{i+1} and D{j+1}: {similarity:.4f}")


# *************************************************************************

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download necessary NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Dataset
documents = [
    "people watch campusx",
    "campusx watch campusx",
    "people write comment",
    "campusx write comment"
]

# Step 1: Preprocessing using NLTK
def preprocess(documents):
    stop_words = set(stopwords.words('english'))
    processed_docs = []

    for doc in documents:
        # Tokenize the document
        tokens = word_tokenize(doc.lower())  # Lowercasing for consistency
        
        # Remove punctuation and stopwords
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        processed_docs.append(" ".join(tokens))
    
    return processed_docs

# Preprocess the documents
processed_documents = preprocess(documents)

# Step 2: TF-IDF Calculation using Scikit-learn
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_documents)

# Convert the TF-IDF matrix to a DataFrame for better visualization
df_tfidf = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(df_tfidf)

# Step 3: Cosine Similarity using SciPy
similarity_matrix = cosine_similarity(tfidf_matrix)

# Display Document Similarities
print("\nDocument Similarities (Cosine):")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        similarity = similarity_matrix[i][j]
        print(f"Similarity between D{i+1} and D{j+1}: {similarity:.4f}")
