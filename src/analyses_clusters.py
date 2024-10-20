import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pointbiserialr
from sklearn.manifold import TSNE
import ast

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def load_data(df_path, embeddings_path):
    print("Loading data...")
    df = pd.read_csv(df_path)

    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)
        
    embeddings = np.array(embeddings)
    return df, embeddings

def topic_distribution(df, num_topics=10, num_words=10):
    print("Performing Topic Distribution Analysis...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['shortened_query'])

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_output = lda_model.fit_transform(doc_term_matrix)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Function to print top words for each topic
    def print_topics(model, feature_names, num_words):
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
            print(f"Topic {topic_idx}: {', '.join(top_words)}")

    print("\nTop words for each topic:")
    print_topics(lda_model, feature_names, num_words)

    topic_names = [f"Topic {i}" for i in range(num_topics)]
    topic_df = pd.DataFrame(lda_output, columns=topic_names)
    topic_df['is_causal'] = df['is_causal']

    causal_dist = topic_df[topic_df['is_causal'] == 1].mean()
    non_causal_dist = topic_df[topic_df['is_causal'] == 0].mean()

    # Calculate the difference in topic distribution
    topic_diff = causal_dist - non_causal_dist

    # Plot the difference in topic distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=topic_names, y=topic_diff[:-1])
    plt.xlabel('Topics')
    plt.ylabel('Difference in Topic Distribution (Causal - Non-Causal)')
    plt.title('Difference in Topic Distribution: Causal vs Non-Causal Questions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('topic_distribution_difference.png')
    plt.close()

    print("\nAnalysis of topic distribution differences:")
    for topic, diff in topic_diff[:-1].items():
        if diff > 0:
            print(f"{topic} is more prevalent in causal questions (difference: {diff:.4f})")
        else:
            print(f"{topic} is more prevalent in non-causal questions (difference: {abs(diff):.4f})")

    print("\nTopic distribution analysis completed. Results saved in 'topic_distribution_difference.png'")

    return lda_model, vectorizer


def named_entity_recognition(df):
    print("Performing Named Entity Recognition...")
    def get_entities(text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    df['entities'] = df['shortened_query'].apply(get_entities)

    causal_entities = [ent for entities in df[df['is_causal'] == 1]['entities'] for ent in entities]
    non_causal_entities = [ent for entities in df[df['is_causal'] == 0]['entities'] for ent in entities]

    causal_entity_types = Counter([ent[1] for ent in causal_entities])
    non_causal_entity_types = Counter([ent[1] for ent in non_causal_entities])

    print("Top 5 entity types in causal questions:")
    print(causal_entity_types.most_common(5))
    print("\nTop 5 entity types in non-causal questions:")
    print(non_causal_entity_types.most_common(5))

def sentiment_analysis(df):
    print("Performing Sentiment Analysis...")
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['shortened_query'].apply(lambda x: sia.polarity_scores(x))
    df['sentiment'] = df['sentiment_scores'].apply(lambda x: x['compound'])

    causal_sentiment = df[df['is_causal'] == 1]['sentiment']
    non_causal_sentiment = df[df['is_causal'] == 0]['sentiment']

    plt.figure(figsize=(10, 6))
    plt.hist(causal_sentiment, bins=20, alpha=0.5, label='Causal')
    plt.hist(non_causal_sentiment, bins=20, alpha=0.5, label='Non-Causal')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution: Causal vs Non-Causal Questions')
    plt.legend()
    plt.savefig('sentiment_distribution.png')
    plt.close()

    print("Sentiment analysis completed. Results saved in 'sentiment_distribution.png'")

def cluster_embeddings(embeddings, n_clusters=5):
    print("Clustering Embeddings...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels

def perform_tsne(embeddings, cache_file='tsne_results.npy'):
    print("Performing t-SNE...")
    if os.path.exists(cache_file):
        print("Loading t-SNE results from cache...")
        tsne_results = np.load(cache_file)
    else:
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)
        print("Saving t-SNE results to cache...")
        np.save(cache_file, tsne_results)
    return tsne_results

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_embeddings(df, tsne_results, cluster_labels):
    print("Visualizing Embeddings...")
    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]
    df['cluster'] = cluster_labels

    # Stratified subsample for better visualization
    stratified_sample = df.groupby('cluster').apply(lambda x: x.sample(frac=0.4)).reset_index(drop=True)

    # Create a diverse color palette
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    color_map = dict(zip(range(n_clusters), colors))

    plt.figure(figsize=(12, 10))
    for cluster in range(n_clusters):
        cluster_data = stratified_sample[stratified_sample['cluster'] == cluster]
        plt.scatter(cluster_data['tsne_x'], cluster_data['tsne_y'], c=[color_map[cluster]], label=f'Cluster {cluster}', s=10)
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.rainbow), label='Cluster')
    plt.title('2D Visualization of Question Embeddings (Colored by Cluster)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('embedding_visualization.png')
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=stratified_sample, x='tsne_x', y='tsne_y', hue='is_causal', palette='Set1', s=10)
    plt.title('2D Visualization of Question Embeddings (Causal vs Non-Causal, Clustered)')
    plt.savefig('embedding_visualization_causal.png')
    plt.close()

    print("Embedding visualization completed. Results saved in 'embedding_visualization.png' and 'embedding_visualization_causal.png'")


def analyze_clusters(df):
    for i in df['cluster'].unique():
        cluster_df = df[df['cluster'] == i]
        causal_ratio = cluster_df['is_causal'].mean()
        print(f"Cluster {i}: {len(cluster_df)} questions, {causal_ratio:.2%} causal")

def correlation_analysis(df, embeddings):
    print("Performing Correlation Analysis...")
    correlations = []
    for i in range(embeddings.shape[1]):
        correlation, p_value = pointbiserialr(df['is_causal'], embeddings[:, i])
        correlations.append((f"dim_{i}", correlation, p_value))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 correlations of embedding dimensions with causal/non-causal label:")
    for dim, corr, p_value in correlations[:10]:
        print(f"{dim}: correlation = {corr:.4f}, p-value = {p_value:.4f}")

    top_dim = int(correlations[0][0].split('_')[1])
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='is_causal', y=embeddings[:, top_dim])
    plt.title(f'Distribution of Top Correlated Dimension for Causal vs Non-Causal Questions')
    plt.savefig('top_dimension_distribution.png')
    plt.close()

    print("Correlation analysis completed. Results saved in 'top_dimension_distribution.png'")

def analyze_clusters_with_examples(df, n_examples=15):
    print("\nAnalyzing clusters with examples:")
    for i in df['cluster'].unique():
        cluster_df = df[df['cluster'] == i]
        causal_ratio = cluster_df['is_causal'].mean()
        print(f"\nCluster {i}: {len(cluster_df)} questions, {causal_ratio:.2%} causal")
        
        print("Causal examples:")
        causal_examples = cluster_df[cluster_df['is_causal'] == 1]['shortened_query'].sample(min(n_examples, sum(cluster_df['is_causal']))).tolist()
        for j, example in enumerate(causal_examples, 1):
            print(f"  {j}. {example}")
        
        print("Non-causal examples:")
        non_causal_examples = cluster_df[cluster_df['is_causal'] == 0]['shortened_query'].sample(min(n_examples, sum(~cluster_df['is_causal']))).tolist()
        for j, example in enumerate(non_causal_examples, 1):
            print(f"  {j}. {example}")

def analyze_causal_vs_noncausal(df):
    print("\nAnalyzing differences between causal and non-causal questions:")
    
    # Length analysis
    df['query_length'] = df['shortened_query'].str.len()
    causal_length = df[df['is_causal'] == 1]['query_length']
    non_causal_length = df[df['is_causal'] == 0]['query_length']
    
    print(f"Average length of causal questions: {causal_length.mean():.2f} characters")
    print(f"Average length of non-causal questions: {non_causal_length.mean():.2f} characters")
    
    # Word count analysis
    df['word_count'] = df['shortened_query'].str.split().str.len()
    causal_words = df[df['is_causal'] == 1]['word_count']
    non_causal_words = df[df['is_causal'] == 0]['word_count']
    
    print(f"Average word count of causal questions: {causal_words.mean():.2f} words")
    print(f"Average word count of non-causal questions: {non_causal_words.mean():.2f} words")
    
    # Most common words
    def get_common_words(text_series, n=10):
        all_words = ' '.join(text_series).lower().split()
        return Counter(all_words).most_common(n)
    
    print("\nMost common words in causal questions:")
    print(get_common_words(df[df['is_causal'] == 1]['shortened_query']))
    
    print("\nMost common words in non-causal questions:")
    print(get_common_words(df[df['is_causal'] == 0]['shortened_query']))

def main(file_path, embeddings_path):
    df, embeddings = load_data(file_path, embeddings_path)
    # if tsne_results.npy exists, load it
    if os.path.exists('tsne_results.npy') and os.path.exists('cluster_labels.npy'):
        tsne_results = np.load('tsne_results.npy')
        cluster_labels = np.load('cluster_labels.npy')
    else:
        cluster_labels = cluster_embeddings(embeddings)
        df['cluster'] = cluster_labels
        tsne_results = perform_tsne(embeddings) 
        # save the results of tsne
        np.save('tsne_results.npy', tsne_results)
        # save the cluster labels in a file
        np.save('cluster_labels.npy', cluster_labels)
    
    visualize_embeddings(df, tsne_results, cluster_labels)
    analyze_clusters(df)
    analyze_clusters_with_examples(df)
    correlation_analysis(df, embeddings)
    
    lda_model, vectorizer = topic_distribution(df)
    
    df['dominant_topic'] = lda_model.transform(vectorizer.transform(df['shortened_query'])).argmax(axis=1)
 
    named_entity_recognition(df)
    sentiment_analysis(df)
    
    analyze_causal_vs_noncausal(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze the CausalQuest dataset")
    parser.add_argument("df_path", type=str, help="Path to the dataset file")
    parser.add_argument("embeddings_path", type=str, help="Path to the embeddings file")
    args = parser.parse_args()
    df_path = args.df_path
    embeddings_path = args.embeddings

    main(df_path, embeddings_path)






















