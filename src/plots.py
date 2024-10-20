import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import pandas as pd


def visualize_embeddings(df, tsne_results, cluster_labels):
    print("Visualizing Embeddings...")
    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]
    df['cluster'] = cluster_labels

    # Stratified subsample for better visualization
    stratified_sample = df.groupby('cluster').apply(lambda x: x.sample(frac=0.3)).reset_index(drop=True)

    # Create a diverse color palette
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    color_map = dict(zip(range(n_clusters), colors))

    plt.figure(figsize=(12, 10))
    for cluster in range(n_clusters):
        cluster_data = stratified_sample[stratified_sample['cluster'] == cluster]
        plt.scatter(cluster_data['tsne_x'], cluster_data['tsne_y'], c=[color_map[cluster]], label=f'Cluster {cluster}', s=10)
    
    # add x and y axis labels
    plt.xlabel('t-SNE Dimension 1', fontsize=24)
    plt.ylabel('t-SNE Dimension 2', fontsize=24)

    #plt.title('2D Visualization of Question Embeddings (Colored by Cluster)', fontsize=28)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-large')
    plt.tight_layout()
    plt.savefig('embedding_visualization.png')
    # save it also as a pdf
    plt.savefig('embedding_visualization.pdf')

    plt.close()

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=stratified_sample, x='tsne_x', y='tsne_y', hue='is_causal', palette='Set1', s=10)
    plt.title('2D Visualization of Question Embeddings (Causal vs Non-Causal, Clustered)')
    plt.savefig('embedding_visualization_causal.png')
    plt.close()

    print("Embedding visualization completed. Results saved in 'embedding_visualization.png' and 'embedding_visualization_causal.png'")


def bloom_likert_plot(path_bloom_labels, path_causalquest):
    """
    Function to create a Bloom Taxonomy plot - causal vs non-causal questions
    """

    df = pd.read_json(path_bloom_labels, lines=True)

    df = pd.concat([df, pd.json_normalize(df['classification'])], axis=1)
    causal_df = pd.read_csv(path_causalquest)
    # match on query_id
    df = pd.merge(df, causal_df, on='query_id', how='left')
    non_causal = df[df['is_causal'] == False]['category'].value_counts(normalize=True) * 100
    causal = df[df['is_causal'] == True]['category'].value_counts(normalize=True) * 100



    # Data
    categories = ['Causal questions', 'Non causal questions']
    remember = [causal['Remembering'], non_causal['Remembering']]
    understand = [causal['Understanding'], non_causal['Understanding']]
    apply = [causal['Applying'], non_causal['Applying']]
    analyze = [causal['Analyzing'], non_causal['Analyzing']]
    evaluate = [causal['Evaluating'], non_causal['Evaluating']]
    create = [causal['Creating'], non_causal['Creating']]

    # Set up the figure and axis
    # put the default font choices in the rcParams dictionary
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']



    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    fig, ax = plt.subplots(figsize=(7, 5))  # Adjusted figure size for two-column layout

    # New diverse color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    # Create the stacked bars
    left = np.zeros(2)
    for i, (data, label, color) in enumerate(zip([remember, understand, apply, analyze, evaluate, create],
                                                ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'],
                                                colors)):
        ax.barh(categories, data, left=left, label=label, color=color, edgecolor='white', height=0.5)
        left += data

    # Customize the chart
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add percentage labels on the bars
    for i, category in enumerate(categories):
        left = 0
        for j, value in enumerate([remember[i], understand[i], apply[i], analyze[i], evaluate[i], create[i]]):
            if value > 7:  # Only add label if the value is greater than 5
                ax.text(left + value/2, i, f'{value:.1f}\%', va='center', ha='center', color='black', fontsize=9)
            left += value
    # Adjust y-axis to reduce white space
    ax.set_ylim(-0.5, 1.5)

    # Add x-axis label
    ax.set_xlabel('Percentage', fontsize=11)

    # Adjust layout and display the chart
    plt.tight_layout()

    # Save it as a pdf
    plt.savefig('bloom_taxonomy.pdf', bbox_inches='tight', dpi=300)

    plt.close(fig)  # Close the figure to free up memory

    print("Bloom taxonomy plot completed. Results saved in 'bloom_taxonomy.pdf'")


def plot_needs_across_sources(path_causalquest, path_needs_labels): 

    causal_df = pd.read_csv(path_causalquest)

    # load question_classifications_user_needs_batch_result.jsonl
    df = pd.read_json(path_needs_labels, lines=True)

    # match on query_id
    df = pd.merge(df, causal_df, on='query_id', how='left')

    # Define the source aggregation
    source_aggregation = {
        'quora': "H-to-H",
        'naturalQuestions': "H-to-SEs",
        'msmarco': "H-to-SEs",
        'wildChat': "H-to-LLMs",
        'ShareGPT': "H-to-LLMs"
    }

    # Apply the aggregation to the DataFrame
    df['aggregated_source'] = df['source'].map(source_aggregation)

    # Set up LaTeX-friendly plot settings
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Use a white background style
    plt.style.use('seaborn-v0_8-white')

    colors = ['#45B7D1', '#FF6B6B', '#4ECDC4', '#F7DC6F', '#FFA07A']

    # Pivot the data to get it in the right format for stacking
    pivot_df = df.pivot_table(index='aggregated_source', columns='category', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # Sort columns by their mean values (descending order)
    column_order = pivot_df.mean().sort_values(ascending=False).index
    pivot_df = pivot_df[column_order]

    # order the pivot_df by ['H-to-SEs', 'H-to-H', 'H-to-LLMs']

    pivot_df = pivot_df.reindex(['H-to-LLMs', 'H-to-H', 'H-to-SEs'  ])

    # Create the horizontal stacked bar chart
    fig, ax = plt.subplots(figsize=(7, 4))  # Adjust height based on number of categories

    left = np.zeros(len(pivot_df))
    for i, column in enumerate(pivot_df.columns):
        ax.barh(pivot_df.index, pivot_df[column], left=left, label=column, color=colors[i % len(colors)], alpha=0.8)
        left += pivot_df[column]

    FONTSIZE = 17  # Base font size
    #ax.set_xlabel('Percentage', fontsize=FONTSIZE+2, fontweight='bold')
    ax.set_xlim(0, 100)

    # Adjust y-axis labels for readability
    plt.yticks(fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add percentage labels on the bars (only for 5% and above)
    for c in ax.containers:
        labels = [f'{v:.1f}\%' if v >= 8 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=FONTSIZE-4, fontweight='bold')

    # Position the legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=FONTSIZE-2, handlelength=1, columnspacing=1)

    plt.tight_layout()
    plt.savefig('needs_across_sources.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free up memory

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plotting functions')
    parser.add_argument('path_bloom_labels', type=str, help='Path to the Bloom labels')
    parser.add_argument('path_causalquest', type=str, help='Path to the CausalQuest dataset')
    parser.add_argument('path_tsne_results', type=str, help='Path to the t-SNE results')
    parser.add_argument('path_cluster_labels', type=str, help='Path to the cluster labels')
    args = parser.parse_args()
    path_bloom_labels = args.path_bloom_labels
    path_causalquest = args.path_causalquest
    path_tsne_results = args.path_tsne_results
    path_cluster_labels = args.path_cluster_labels
    
    
    bloom_likert_plot(path_bloom_labels, path_causalquest)
    df = pd.read_csv(path_causalquest)
    tsne_results = np.load(path_tsne_results)
    cluster_labels = np.load(path_cluster_labels)

    visualize_embeddings(df, tsne_results, cluster_labels)
