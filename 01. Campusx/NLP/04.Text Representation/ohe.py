

import numpy as np
import pandas as pd
from collections import Counter

# 1. Corpus and Documents
documents = [
    "people watch campusx",
    "campusx watch campusx",
    "people write comment",
    "campusx write comment"
]

# Print the documents
print("Documents:")
for i, doc in enumerate(documents):
    print(f"D{i+1}: {doc}")

# 2. Create the corpus by combining all documents
corpus = " ".join(documents)
print("\nCorpus:")
print(corpus)

# 3. Create vocabulary (unique words)
def create_vocabulary(corpus):
    # Split the corpus into words and get unique words
    words = corpus.split()
    vocabulary = sorted(list(set(words)))
    return vocabulary

vocabulary = create_vocabulary(corpus)
V = len(vocabulary)

print("\nVocabulary:", vocabulary)
print(f"Vocabulary Size (V): {V}")

# 4. One-Hot Encoding Vectors
def create_one_hot_encoding(word, vocabulary):
    # Create a vector of zeros with length equal to vocabulary size
    vector = [0] * len(vocabulary)
    
    # Set the position corresponding to the word to 1
    if word in vocabulary:
        word_index = vocabulary.index(word)
        vector[word_index] = 1
    
    return vector

# Generate one-hot encodings for each word in vocabulary
one_hot_encodings = {}
for word in vocabulary:
    one_hot_encodings[word] = create_one_hot_encoding(word, vocabulary)

# Print one-hot encodings
print("\nOne-Hot Encoding Vectors:")
for word, vector in one_hot_encodings.items():
    print(f"{word}: {vector}")

# 5. Document Vector Representation
def encode_document(doc, vocabulary):
    words = doc.split()
    encoded_doc = []
    
    for word in words:
        if word in vocabulary:
            encoded_doc.append(create_one_hot_encoding(word, vocabulary))
    
    return np.array(encoded_doc)

# Encode each document
encoded_documents = []
for i, doc in enumerate(documents):
    encoded_doc = encode_document(doc, vocabulary)
    encoded_documents.append(encoded_doc)
    
    print(f"\nDocument D{i+1} One-Hot Encoding:")
    print(encoded_doc)
    print(f"Shape: {encoded_doc.shape}")

# 6. Demonstrate OOV Handling
def handle_oov_document(doc, vocabulary):
    words = doc.split()
    known_words = []
    unknown_words = []
    
    for word in words:
        if word in vocabulary:
            known_words.append(word)
        else:
            unknown_words.append(word)
    
    return known_words, unknown_words

# Example of handling OOV words
print("\n7. Handling Out-of-Vocabulary (OOV) Words:")
new_doc = "hello campusx peoples"
known, unknown = handle_oov_document(new_doc, vocabulary)

print(f"New Document: '{new_doc}'")
print(f"Known Words: {known}")
print(f"Unknown Words (OOV): {unknown}")

# Try to encode the document with OOV words
encoded_new_doc = encode_document(new_doc, vocabulary)
print("\nEncoded New Document (only known words are encoded):")
print(encoded_new_doc)
print(f"Shape: {encoded_new_doc.shape}")

# 8. Demonstrate Orthogonality - Calculate distances between word vectors
print("\n8. Demonstrating lack of semantic relationships:")

# Create a simple example with three words
simple_vocabulary = ["walk", "run", "shoe"]
simple_encodings = {}

for word in simple_vocabulary:
    vector = [0] * len(simple_vocabulary)
    vector[simple_vocabulary.index(word)] = 1
    simple_encodings[word] = vector

print("Simple One-Hot Encodings:")
for word, vector in simple_encodings.items():
    print(f"{word}: {vector}")

# Calculate Euclidean distances between vectors
def euclidean_distance(vec1, vec2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

print("\nEuclidean Distances:")
for word1 in simple_vocabulary:
    for word2 in simple_vocabulary:
        if word1 != word2:
            dist = euclidean_distance(simple_encodings[word1], simple_encodings[word2])
            print(f"Distance between '{word1}' and '{word2}': {dist:.2f}")


#************************************************************************************


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# # 1. Corpus and Documents
# documents = [
#     "people watch campusx",
#     "campusx watch campusx",
#     "people write comment",
#     "campusx write comment"
# ]

# # Print the documents
# print("Documents:")
# for i, doc in enumerate(documents):
#     print(f"D{i+1}: {doc}")

# # 2. Create the corpus by combining all documents
# corpus = " ".join(documents)
# print("\nCorpus: ")
# print(corpus)

# # 3. Tokenize and Create Vocabulary (unique words)
# def tokenize(corpus):
#     # Tokenize the corpus by splitting on whitespace
#     words = corpus.split()
#     return words

# words = tokenize(corpus)
# vocabulary = sorted(list(set(words)))  # Unique sorted words

# print("\nVocabulary: ", vocabulary)

# # 4. One-Hot Encoding using `OneHotEncoder` from `scikit-learn`
# # Reshape the words list into a 2D array for OneHotEncoder (since it expects 2D input)
# encoder = OneHotEncoder(sparse=False)

# # Fit the encoder to the vocabulary (the words list as 2D array)
# encoded_vocabulary = encoder.fit_transform(np.array(vocabulary).reshape(-1, 1))

# # Convert the result to a DataFrame for better readability
# encoded_vocabulary_df = pd.DataFrame(encoded_vocabulary, columns=vocabulary)
# print("\nOne-Hot Encoding Vectors:")
# print(encoded_vocabulary_df)

# # 5. Document Vector Representation
# def encode_document(doc, vocabulary, encoder):
#     # Tokenize the document and encode each word using the OneHotEncoder
#     words = doc.split()
#     encoded_doc = []
    
#     for word in words:
#         if word in vocabulary:
#             encoded_word = encoder.transform([[word]])
#             encoded_doc.append(encoded_word)
    
#     return np.array(encoded_doc).squeeze()

# # Encode each document
# encoded_documents = []
# for i, doc in enumerate(documents):
#     encoded_doc = encode_document(doc, vocabulary, encoder)
#     encoded_documents.append(encoded_doc)
    
#     print(f"\nDocument D{i+1} One-Hot Encoding:")
#     print(encoded_doc)
#     print(f"Shape: {encoded_doc.shape}")

# # 6. Handling Out-of-Vocabulary (OOV) Words
# def handle_oov_document(doc, vocabulary):
#     words = doc.split()
#     known_words = []
#     unknown_words = []
    
#     for word in words:
#         if word in vocabulary:
#             known_words.append(word)
#         else:
#             unknown_words.append(word)
    
#     return known_words, unknown_words

# # Example of handling OOV words
# print("\n7. Handling Out-of-Vocabulary (OOV) Words:")
# new_doc = "hello campusx peoples"
# known, unknown = handle_oov_document(new_doc, vocabulary)

# print(f"New Document: '{new_doc}'")
# print(f"Known Words: {known}")
# print(f"Unknown Words (OOV): {unknown}")

# # Encode the new document (only known words)
# encoded_new_doc = encode_document(new_doc, vocabulary, encoder)
# print("\nEncoded New Document (only known words are encoded):")
# print(encoded_new_doc)
# print(f"Shape: {encoded_new_doc.shape}")


# **************************************************************

# import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical

# # 1. Corpus and Documents
# documents = [
#     "people watch campusx",
#     "campusx watch campusx",
#     "people write comment",
#     "campusx write comment"
# ]

# # Print the documents
# print("Documents:")
# for i, doc in enumerate(documents):
#     print(f"D{i+1}: {doc}")

# # 2. Create Tokenizer and Fit on Corpus
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(documents)

# # 3. Create Vocabulary and Word Index Mapping
# vocabulary = tokenizer.word_index  # Dictionary of word to index
# vocab_size = len(vocabulary) + 1  # +1 to account for padding
# print("\nVocabulary (Word to Index Mapping):", vocabulary)

# # 4. Convert Documents to Sequences (of indices)
# sequences = tokenizer.texts_to_sequences(documents)
# print("\nSequences (of word indices):", sequences)

# # 5. One-Hot Encoding using `to_categorical`
# # One-Hot Encoding the sequences
# one_hot_encoded_documents = [to_categorical(seq, num_classes=vocab_size) for seq in sequences]

# # Print One-Hot Encoded Documents
# print("\nOne-Hot Encoded Documents:")
# for i, encoded_doc in enumerate(one_hot_encoded_documents):
#     print(f"Document {i+1}:")
#     print(encoded_doc)
#     print(f"Shape: {encoded_doc.shape}")

# # 6. Handling Out-of-Vocabulary (OOV) Words
# # New document with OOV words
# new_doc = "hello campusx peoples"

# # Convert the new document to a sequence of indices
# new_doc_sequence = tokenizer.texts_to_sequences([new_doc])[0]
# print("\nNew Document (OOV words):", new_doc_sequence)

# # Convert the sequence to One-Hot Encoding (only known words will be encoded)
# new_doc_one_hot = to_categorical(new_doc_sequence, num_classes=vocab_size)
# print("\nOne-Hot Encoding for New Document:")
# print(new_doc_one_hot)
# print(f"Shape: {new_doc_one_hot.shape}")





# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from collections import Counter

# # Download NLTK punkt tokenizer models (only once)
# nltk.download('punkt')

# # 1. Corpus and Documents
# documents = [
#     "people watch campusx",
#     "campusx watch campusx",
#     "people write comment",
#     "campusx write comment"
# ]

# # Print the documents
# print("Documents:")
# for i, doc in enumerate(documents):
#     print(f"D{i+1}: {doc}")

# # 2. Tokenize the documents
# tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
# print("\nTokenized Documents:")
# for i, doc in enumerate(tokenized_docs):
#     print(f"D{i+1}: {doc}")

# # 3. Create Vocabulary (Unique words)
# # Flatten the tokenized words and get unique words
# all_words = [word for doc in tokenized_docs for word in doc]
# vocabulary = sorted(set(all_words))
# vocab_size = len(vocabulary)
# print("\nVocabulary:", vocabulary)
# print(f"Vocabulary Size: {vocab_size}")

# # 4. One-Hot Encoding
# def create_one_hot_encoding(word, vocabulary):
#     # Create a vector of zeros with length equal to vocabulary size
#     vector = [0] * len(vocabulary)
    
#     # Set the position corresponding to the word to 1
#     if word in vocabulary:
#         word_index = vocabulary.index(word)
#         vector[word_index] = 1
    
#     return vector

# # Create one-hot encodings for each word in the vocabulary
# one_hot_encodings = {word: create_one_hot_encoding(word, vocabulary) for word in vocabulary}

# # Print one-hot encodings
# print("\nOne-Hot Encoding Vectors:")
# for word, vector in one_hot_encodings.items():
#     print(f"{word}: {vector}")

# # 5. Document Vector Representation
# def encode_document(doc, vocabulary):
#     # Tokenize the document
#     words = word_tokenize(doc.lower())
    
#     # Encode each word and create a list of one-hot encoded vectors
#     encoded_doc = [create_one_hot_encoding(word, vocabulary) for word in words if word in vocabulary]
    
#     return np.array(encoded_doc)

# # Encode each document and print the result
# encoded_documents = []
# for i, doc in enumerate(documents):
#     encoded_doc = encode_document(doc, vocabulary)
#     encoded_documents.append(encoded_doc)
    
#     print(f"\nDocument D{i+1} One-Hot Encoding:")
#     print(encoded_doc)
#     print(f"Shape: {encoded_doc.shape}")

# # 6. Handling Out-of-Vocabulary (OOV) Words
# new_doc = "hello campusx peoples"
# new_doc_tokenized = word_tokenize(new_doc.lower())
# print("\nNew Document Tokenized:", new_doc_tokenized)

# # Only encode known words from the vocabulary
# encoded_new_doc = [create_one_hot_encoding(word, vocabulary) for word in new_doc_tokenized if word in vocabulary]
# print("\nEncoded New Document (only known words are encoded):")
# print(np.array(encoded_new_doc))



# import torch
# import nltk
# from nltk.tokenize import word_tokenize

# # Download NLTK punkt tokenizer models (only once)
# nltk.download('punkt')

# # 1. Corpus and Documents
# documents = [
#     "people watch campusx",
#     "campusx watch campusx",
#     "people write comment",
#     "campusx write comment"
# ]

# # Print the documents
# print("Documents:")
# for i, doc in enumerate(documents):
#     print(f"D{i+1}: {doc}")

# # 2. Tokenize the documents
# tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
# print("\nTokenized Documents:")
# for i, doc in enumerate(tokenized_docs):
#     print(f"D{i+1}: {doc}")

# # 3. Create Vocabulary (Unique words)
# # Flatten the tokenized words and get unique words
# all_words = [word for doc in tokenized_docs for word in doc]
# vocabulary = sorted(set(all_words))
# vocab_size = len(vocabulary)
# print("\nVocabulary:", vocabulary)
# print(f"Vocabulary Size: {vocab_size}")

# # 4. One-Hot Encoding using PyTorch
# def create_one_hot_encoding(word, vocabulary):
#     # Create a tensor of zeros with size equal to the vocabulary size
#     vector = torch.zeros(vocab_size)
    
#     # Set the position corresponding to the word to 1
#     if word in vocabulary:
#         word_index = vocabulary.index(word)
#         vector[word_index] = 1
    
#     return vector

# # Create one-hot encodings for each word in the vocabulary
# one_hot_encodings = {word: create_one_hot_encoding(word, vocabulary) for word in vocabulary}

# # Print one-hot encodings
# print("\nOne-Hot Encoding Vectors:")
# for word, vector in one_hot_encodings.items():
#     print(f"{word}: {vector}")

# # 5. Document Vector Representation using PyTorch
# def encode_document(doc, vocabulary):
#     # Tokenize the document
#     words = word_tokenize(doc.lower())
    
#     # Encode each word and create a list of one-hot encoded tensors
#     encoded_doc = [create_one_hot_encoding(word, vocabulary) for word in words if word in vocabulary]
    
#     return torch.stack(encoded_doc)  # Stack tensors into a tensor for the document

# # Encode each document and print the result
# encoded_documents = []
# for i, doc in enumerate(documents):
#     encoded_doc = encode_document(doc, vocabulary)
#     encoded_documents.append(encoded_doc)
    
#     print(f"\nDocument D{i+1} One-Hot Encoding:")
#     print(encoded_doc)
#     print(f"Shape: {encoded_doc.shape}")

# # 6. Handling Out-of-Vocabulary (OOV) Words
# new_doc = "hello campusx peoples"
# new_doc_tokenized = word_tokenize(new_doc.lower())
# print("\nNew Document Tokenized:", new_doc_tokenized)

# # Only encode known words from the vocabulary
# encoded_new_doc = [create_one_hot_encoding(word, vocabulary) for word in new_doc_tokenized if word in vocabulary]
# encoded_new_doc = torch.stack(encoded_new_doc) if encoded_new_doc else torch.empty(0)
# print("\nEncoded New Document (only known words are encoded):")
# print(encoded_new_doc)
# print(f"Shape: {encoded_new_doc.shape}")
