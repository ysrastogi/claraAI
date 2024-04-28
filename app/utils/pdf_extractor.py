import re
import torch
from PyPDF2 import PdfReader
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing
import camelot


def process_text_file_and_store_vectors(file_paths, percentile, vectorstore_path):
    sop = ''
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    sop += page.extract_text()
                # Remove the .lower() here, as it's causing the error
                tables = camelot.read_pdf(file_path)
                for table in tables:
                    df = table.df
                    latex_table = df.to_latex(index=False)
                    sop += latex_table
        else:
            with open(file_path, 'r') as f:
                sop = f.read()
        single_sentences_list = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\t])', sop)
        sentences = [{'sentence': x, 'index':i} for i, x in enumerate(single_sentences_list)]

    def combine_sentences(sentences, buffer_size=1):
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - buffer_size, i):
                if j>=0:
                    combined_sentence += sentences[j]['sentence'] + ' '
            combined_sentence += sentences[i]['sentence']

            for j in range(i+1, i+1+ buffer_size):
                if j< len(sentences):
                    combined_sentence += ' '+ sentences[j]['sentence']
            sentences[i]['combined_sentence'] = combined_sentence
        return sentences
    sentences = combine_sentences(sentences) 
    print(sentences)

    def build_bert_embeddings(sentences):
        # Initialize BERT model and tokenizer
        model_name = 'bert-base-uncased'  # Replace with the specific BERT model you want to use
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        # Check if MPS device is available
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            print("Using MPS device:", mps_device)
        else:
            print("MPS device not found.")
            # If MPS device is not available, fallback to GPU or CPU
            mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(mps_device)

        print("Building embeddings")
        bert_embeddings = []
        for i, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            input_ids = tokenizer.encode(sentence['combined_sentence'], add_special_tokens=True, return_tensors='pt', max_length = 512, truncation=True).to(mps_device)
            with torch.no_grad():
                outputs = model(input_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling to get a single vector for the entire sentence
                bert_embeddings.append(embeddings.cpu().numpy())  # Move tensor to CPU and then convert to NumPy array


        return bert_embeddings 
    
    embeddings = build_bert_embeddings(sentences)

    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    def calculate_cosine_distances(sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding_current, embedding_next)[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance


        return distances, sentences

    distances, sentences = calculate_cosine_distances(sentences)

    def get_chunks_based_on_percentile(sentences, distances, percentile=94):
        # We need to get the distance threshold that we'll consider an outlier
        # We'll use numpy .percentile() for this
        breakpoint_distance_threshold = np.percentile(distances, percentile) # If you want more chunks, lower the percentile cutoff

        # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

        # Initialize the start index
        start_index = 0
        print("chunking")
        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            
            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        # chunks now contains the chunked sentences
        return chunks


    chunks = get_chunks_based_on_percentile(sentences, distances, percentile)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device":"cpu"}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    print("Saving Embeddings")
    print("Saving Embeddings")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(vectorstore_path)

# file_paths = ['Support SLA Policy.pdf', 'SOP for Tech Issues.txt']
# percentile = 96
# vectorstore_path = 'vectorstore.faiss'
# process_text_file_and_store_vectors(file_paths, percentile, vectorstore_path)