import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.manifold import TSNE
import json

# Load the OpenAI client
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small", row_num=None):
    text = text.replace("\n", " ")
    print(f"Generating embedding for row {row_num}: {text}")
    print()
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def generate_embeddings_and_tsne(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # add a column with numbers from 0 to the number of rows
    df['row_num'] = range(len(df))


    # Generate embeddings
    print("Generating embeddings...")
    embeddings = df.apply(lambda x: get_embedding(x['shortened_query'], row_num=x['row_num']), axis=1).tolist()

    # save the embeddings to a file
    with open("causalquest_embeddings.json", 'w') as f: # file that contains only the embeddings 
        json.dump(embeddings, f)


    # Add embeddings to the dataframe
    df['ada_embedding'] = embeddings
    
    
    # Convert embeddings to a numpy array
    embeddings = np.array(df['ada_embedding'].tolist())
    
    # Apply tsne
    print("Applying tsne...")
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)

    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Save embeddings, tsne coordinates, and other necessary data
    output_data = {
        'query_id': df['query_id'].tolist(),
        'query': df['query'].tolist(),
        'shortened_query': df['shortened_query'].tolist(),
        'source': df['source'].tolist(),
        'is_causal': df['is_causal'].tolist(),
        'domain_class': df['domain_class'].tolist(),
        'action_class': df['action_class'].tolist(),
        'is_subjective': df['is_subjective'].tolist(),
        'tsne_x': tsne_embeddings[:, 0].tolist(),
        'tsne_y': tsne_embeddings[:, 1].tolist()
    }
    
    print(f"Saving data to {output_file}...")
    with open(output_file, 'w') as f: # file that contains the embeddings and the tsne coordinates
        json.dump(output_data, f)


    #on another file, save also the embeddings
    output_file = "causalquest_w_embeddings.csv" # file that contains the embeddings and the original data
    print(f"Saving data to {output_file}...")
    df.to_csv(output_file)

    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate embeddings and tsne for a dataset')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file


    generate_embeddings_and_tsne(input_file, output_file)