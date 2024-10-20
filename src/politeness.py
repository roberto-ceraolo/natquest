import pandas as pd
import spacy
import nltk
from convokit import PolitenessStrategies, Corpus
from convokit import textParser
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
spacy.load('en_core_web_sm')

def generate_politeness_dataset(causalquest_path):
    
    causalquest = pd.read_csv(causalquest_path)

    # rename query_id to ID
    causalquest.rename(columns={'query_id': 'id'}, inplace=True)

    # rename shortened_query to text
    causalquest.rename(columns={'shortened_query': 'text'}, inplace=True)

    # add a timestamp in format like 1332788704
    causalquest['timestamp'] = pd.to_datetime('2024-01-01')

    # add a speaker column
    causalquest['speaker'] = 'user'

    # add a reply_to column
    causalquest['reply_to'] = 'None'

    # add a conversation_id column
    causalquest['conversation_id'] = 'None'

    # drop all the columns that are not needed
    causalquest = causalquest[['id', 'timestamp', 'text', 'speaker', 'reply_to', 'conversation_id', 'source']]


    new_corpus = Corpus.from_pandas(causalquest)


    parser = textParser.TextParser()

    new_corpus = parser.transform(new_corpus)

    ps = PolitenessStrategies(verbose=1000)
    new_corpus = ps.transform(new_corpus)


    utterance_ids = new_corpus.get_utterance_ids()
    rows = []
    for uid in utterance_ids:
        query_id = {"id": uid}
        utt = new_corpus.get_utterance(uid)
        row = utt.meta["politeness_strategies"]
        row.update(query_id)
        rows.append(row)
    politeness_strategies = pd.DataFrame(rows, index=utterance_ids)
    politeness_strategies['id'] = politeness_strategies['id'].astype('int64')

    causalquest_merged = causalquest.merge(politeness_strategies, on='id')
    return causalquest_merged

def politeness_barchart(df):

    # Define the color palette
    colors = ['#FF6B6B', '#4ECDC4','#F7DC6F', '#FFA07A', '#45B7D1', '#98D8C8']

    # 1. Calculate mean politeness features for each source
    politeness_cols = [col for col in df.columns if col.startswith('feature_politeness_')]
    df['overall_politeness'] = df[politeness_cols].sum(axis=1)

    politeness_by_source = df.groupby('source')[politeness_cols + ['overall_politeness']].mean()


    # Sort the data by overall_politeness in descending order
    politeness_by_source_sorted = politeness_by_source.sort_values('overall_politeness', ascending=False)
    politeness_by_source_sorted.index = politeness_by_source_sorted.index.map({'quora': 'Quora', 'wildChat': 'WildChat', 'naturalQuestions': 'NaturalQuestions', 'msmarco': 'MSMARCO', 'ShareGPT'  : 'ShareGPT'})

    # 2. Plot the overall politeness score by source
    plt.figure(figsize=(12, 7))

    # Remove the grid
    plt.grid(False)

    bars = plt.bar(politeness_by_source_sorted.index, 
                politeness_by_source_sorted['overall_politeness'], 
                color=colors[:len(politeness_by_source_sorted)])

    plt.xlabel('Source', fontsize=20)
    plt.ylabel('Mean Politeness Score', fontsize=20)
    #plt.title('Mean Politeness Score by Source', fontsize=16)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=15)

    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=45, ha='right', fontsize=15)

    # Adjust layout and display the plot
    plt.tight_layout()

    # save it as a pdf
    plt.savefig('politeness_by_source.pdf')

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize the politeness strategies in the causal question dataset')
    parser.add_argument('causalquest_path', type=str, help='Path to the causal question dataset')
    args = parser.parse_args()
    causalquest_path = args.causalquest_path

    df_politeness = generate_politeness_dataset(causalquest_path)
    politeness_barchart(df_politeness)

