import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

import clip
from matplotlib import pyplot as plt

import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_file, export_png
from bokeh.palettes import Category10_10  # For color palette

pd.options.mode.chained_assignment = None  # default='warn'

csv_path = "data/LQ Survey.csv"

def keep_only_clean(row):
    row = [r for r in row if not len(r) < 5] # Clear empty rows / only-score entries
    if len(row) > 0:
        if not row[0][-1].isnumeric() or not len(row[0]) >= 2:
            row = None
    else:
        row = None
    return row    

def prompts_to_list(row):
    row_split = row.split('\n')
    return row_split

def remove_bad_entries(responses, keep_only_with_aux=True, only_clean_rows=True):
    if keep_only_with_aux:
        responses = responses.dropna(how="any")
    # else:
    #     responses = responses[responses['have_been'].notna()]
    responses['descriptions'] = responses['descriptions'].apply(prompts_to_list)   
    if only_clean_rows:
        responses['descriptions'] = responses['descriptions'].apply(keep_only_clean)
    responses = responses[responses['descriptions'].notna()]
    lists_of_prompts = responses['descriptions'].values
    return responses

def plot_clusters(clusters_df):
    ## PLOT AS SCATTERPLOT ##
    output_file("scatterplot.html") 
    p = figure(title = "Clusters")

    # Get unique cluster numbers
    unique_clusters = clusters_df['cl'].unique()
    num_clusters = len(unique_clusters)

    # Define a color palette for the clusters
    colors = Category10_10[:num_clusters]
    cluster_color_map = dict(zip(unique_clusters, colors))

    # Iterate over the data and add scatter markers
    for cluster, color in cluster_color_map.items():
        cluster_data = clusters_df[clusters_df['cl'] == cluster]
        p.scatter(x=cluster_data['x'], y=cluster_data['y'], color=color, legend_label=f'Cluster {cluster}')

    # Add plot title and axes labels
    p.title.text = 'Cluster Scatter Plot'
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'Y'

    # Add legend
    p.legend.title = 'Clusters'
    show(p)

responses = pd.read_csv(csv_path)
responses = responses.drop('Tijdstempel', axis=1)
responses.columns.values[0] = 'have_been'
responses.columns.values[1] = 'confidence'
responses.columns.values[2] = 'descriptions'

filtered_info = remove_bad_entries(responses, keep_only_with_aux=False)
import numpy as np
stats_frame = filtered_info.dropna()
mean_conf = np.mean(stats_frame['confidence'])


# Split prompts / values
prompt_lists = filtered_info['descriptions'].to_list()
lengths = [len(l) for l in prompt_lists]
all_prompts = [item for sublist in prompt_lists for item in sublist]

user_index = []
have_been = []
conf = []
values_in_order = []
prompts_in_order = []
for user_id, user_prompts in enumerate(prompt_lists):
    for p in user_prompts:
        # Get metadata
        user_index.append(user_id)
        have_been.append(filtered_info.iloc[user_id]['have_been'])
        conf.append(filtered_info.iloc[user_id]['confidence'])

        # Split prompt values
        prompt_score_pairs = p.split(':')
        prompts_in_order.append(prompt_score_pairs[0])
        values_in_order.append(int(prompt_score_pairs[1].strip()))
prompts_df = pd.DataFrame({'user_index': user_index,
                           'have_been': have_been,
                           'confidence': conf,
                           'values_in_order': values_in_order,
                           'prompts_in_order': prompts_in_order})
prompts_df.to_csv("data/prompts_cleaned.csv")

## TODO: Forward pass script, averaging over all images
# Overload base class
# Loop:
#   Re-setup model with different encoded prompts
#   Retrieve preds column from output DF
#   Mean/variance of all SON scores

## Cluster
# Import clip & encode
net_name = "ViT-L/14"
model, _ = clip.load(net_name)

with torch.no_grad():
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_in_order]).to(0)
    encoded_prompts = model.encode_text(tokenized_prompts).cpu().numpy()
# prompt_values = torch.tensor(values).to(0)

# Run t-SNE over embeddings
# tsne_model = TSNE(n_components=2, learning_rate='auto', perplexity=3, init='pca')# learning_rate='auto', init='random', perplexity=3)
# prompts_tsne = tsne_model.fit_transform(encoded_prompts)
# prompts_x = prompts_tsne[:, 0]
# prompts_y = prompts_tsne[:, 1]
pca = PCA(n_components=3).fit_transform(encoded_prompts)
prompts_x = pca[:, 0]
prompts_y = pca[:, 1]

# prompt_plot = plt.scatter(prompts_x, prompts_y)
# plt.savefig('plot.jpg')

# k-Means / DBScan over samples
# clustering = DBSCAN(eps=3, min_samples=2).fit(prompts_tsne)
km = KMeans(n_clusters=25).fit(encoded_prompts)
clusters = km.labels_

## MERGE INTO DATAFRAME ##
clusters_df = pd.DataFrame({'x': prompts_x, 'y': prompts_y, 'cl': clusters})
clusters_df['prompts'] = prompts_in_order
clusters_df['values'] = values_in_order
clusters_df[clusters_df['cl'].isin([0])]

## TODO: MAKE DATAFRAMES FOR EACH USER -> TEST