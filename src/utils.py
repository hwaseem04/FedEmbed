import torch
import os
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = False)
        # label: 0 similar
        # label: 1 dissimilar
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def plot_pca_embeddings(embeds, round_num, model_num, save_dir="pca_plots"):
    # Prepare data for PCA
    all_embeddings = []
    all_labels = []
    all_clients = []

    for emb_id, class_embeddings in enumerate(embeds):
        client_elem_count = 0
        for class_id, embedding in class_embeddings.items():
            embedding = embedding.numpy()
            if embedding.ndim == 1:
                embedding = embedding.reshape(-1, 128)
            
            all_embeddings.append(embedding)
            tmp = [class_id] * len(embedding)
            all_labels.extend(tmp)
            client_elem_count += len(embedding)
        
        tmp = [emb_id] * client_elem_count
        all_clients.extend(tmp)

    # Convert to numpy array
    all_embeddings = np.vstack(all_embeddings)
    print(len(all_embeddings[all_embeddings == np.nan]))
    # Apply PCA
    pca = PCA(n_components=2)
    low_dim_embeddings = pca.fit_transform(all_embeddings)

    # Plot 1: Only client embeddings
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("bright", 10)
    
    if model_num == 2:
        markers = ['o', 's', 'x', 'D', '^']  # Different markers for each client
        L = [2,3,4]
    elif model_num == 4: 
                   #c1, c2, c3, c4 s_p,c1_p, c2_p,c3_p, c4_p
        markers = ['o','s','D','^','x', 'v', '<', '>', 'p']
        L = [4,5,6,7,8]

    for emb_id in range(model_num*2+1):  # Clients 0 and 1 (optionally 2, 3)
        if emb_id in L:
            continue
        indices = [i for i, x in enumerate(all_clients) if x == emb_id]
        subset = low_dim_embeddings[indices]
        client_labels = [all_labels[i] for i in indices]
        sns.scatterplot(x=subset[:, 0], y=subset[:, 1], hue=client_labels, palette=palette, legend=None, marker=markers[emb_id], s=40, alpha=0.9)

    # Custom legend for client IDs
    from matplotlib.lines import Line2D
    legend_elements_clients = [Line2D([0], [0], marker=markers[i], color='w', label=f'Client {i}', markerfacecolor='k', markersize=10) for i in range(model_num*2+1) if i not in L]
    legend_elements_classes = [Line2D([0], [0], marker='o', color='w', label=f'Class {i}', markerfacecolor=palette[i], markersize=10) for i in range(10)]
    plt.legend(handles=legend_elements_clients + legend_elements_classes, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'PCA of Client Embeddings - Round {round_num}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'pca_clients_round_{round_num}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Client PCA plot saved to {save_path}")
    
    # Plot 2: Only 'Averaged', 'C0_Prototype', and 'C1_Prototype' embeddings
    plt.figure(figsize=(10, 8))
    for emb_id in L:  # 'Averaged', 'C0_Prototype', 'C1_Prototype'
        indices = [i for i, x in enumerate(all_clients) if x == emb_id]
        subset = low_dim_embeddings[indices]
        client_labels = [all_labels[i] for i in indices]
        s = 160
        sns.scatterplot(x=subset[:, 0], y=subset[:, 1], hue=client_labels, palette=palette, legend=None, marker=markers[emb_id], s=s, alpha=0.9)

    if model_num == 2:
        legend_elements_prototypes = [
            Line2D([0], [0], marker=markers[2], color='w', label='Averaged', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[3], color='w', label='C0_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[4], color='w', label='C1_Prototype', markerfacecolor='k', markersize=10),
        ]
    elif model_num == 4:
        legend_elements_prototypes = [
            Line2D([0], [0], marker=markers[4], color='w', label='Averaged', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[5], color='w', label='C0_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[6], color='w', label='C1_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[7], color='w', label='C2_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[8], color='w', label='C3_Prototype', markerfacecolor='k', markersize=10),
        ]
    plt.legend(handles=legend_elements_prototypes, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'PCA of Prototypes - Round {round_num}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

    save_path = os.path.join(save_dir, f'pca_prototypes_round_{round_num}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Prototypes PCA plot saved to {save_path}")



import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def plot_pca_embeddings(embeds, round_num, model_num, save_dir="pca_plots"):
    # Prepare data for PCA
    all_embeddings = []
    all_labels = []
    all_clients = []

    for emb_id, class_embeddings in enumerate(embeds):
        client_elem_count = 0
        for class_id, embedding in class_embeddings.items():
            embedding = embedding.numpy()
            if embedding.ndim == 1:
                embedding = embedding.reshape(-1, 128)
            
            all_embeddings.append(embedding)
            tmp = [class_id] * len(embedding)
            all_labels.extend(tmp)
            client_elem_count += len(embedding)
        
        tmp = [emb_id] * client_elem_count
        all_clients.extend(tmp)

    # Convert to numpy array
    all_embeddings = np.vstack(all_embeddings)

    # Apply PCA
    pca = PCA(n_components=2)
    low_dim_embeddings = pca.fit_transform(all_embeddings)

    # Plot 1: Only client embeddings
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("bright", 10)
    
    if model_num == 2:
        markers = ['o', 's', 'x', 'D', '^']  # Different markers for each client
        L = [2,3,4]
    elif model_num == 4: 
        markers = ['o','s','D','^','x', 'v', '<', '>', 'p']
        L = [4,5,6,7,8]

    for emb_id in range(model_num*2+1):  # Clients 0 and 1 (optionally 2, 3)
        if emb_id in L:
            continue
        indices = [i for i, x in enumerate(all_clients) if x == emb_id]
        subset = low_dim_embeddings[indices]
        client_labels = [all_labels[i] for i in indices]
        sns.scatterplot(x=subset[:, 0], y=subset[:, 1], hue=client_labels, palette=palette, legend=None, marker=markers[emb_id], s=40, alpha=0.9)

    # Custom legend for client IDs
    from matplotlib.lines import Line2D
    legend_elements_clients = [Line2D([0], [0], marker=markers[i], color='w', label=f'Client {i}', markerfacecolor='k', markersize=10) for i in range(model_num*2+1) if i not in L]
    legend_elements_classes = [Line2D([0], [0], marker='o', color='w', label=f'Class {i}', markerfacecolor=palette[i], markersize=10) for i in range(10)]
    plt.legend(handles=legend_elements_clients + legend_elements_classes, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'PCA of Client Embeddings - Round {round_num}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'pca_clients_round_{round_num}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Client PCA plot saved to {save_path}")

    # Log the first plot to wandb
    wandb.log({
        f"PCA of Client Embeddings - Round {round_num}": wandb.Image(save_path),
        "round": round_num
    })

    # Plot 2: Only 'Averaged', 'C0_Prototype', and 'C1_Prototype' embeddings
    plt.figure(figsize=(10, 8))
    for emb_id in L:  # 'Averaged', 'C0_Prototype', 'C1_Prototype'
        indices = [i for i, x in enumerate(all_clients) if x == emb_id]
        subset = low_dim_embeddings[indices]
        client_labels = [all_labels[i] for i in indices]
        s = 160
        sns.scatterplot(x=subset[:, 0], y=subset[:, 1], hue=client_labels, palette=palette, legend=None, marker=markers[emb_id], s=s, alpha=0.9)

    if model_num == 2:
        legend_elements_prototypes = [
            Line2D([0], [0], marker=markers[2], color='w', label='Averaged', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[3], color='w', label='C0_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[4], color='w', label='C1_Prototype', markerfacecolor='k', markersize=10),
        ]
    elif model_num == 4:
        legend_elements_prototypes = [
            Line2D([0], [0], marker=markers[4], color='w', label='Averaged', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[5], color='w', label='C0_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[6], color='w', label='C1_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[7], color='w', label='C2_Prototype', markerfacecolor='k', markersize=10),
            Line2D([0], [0], marker=markers[8], color='w', label='C3_Prototype', markerfacecolor='k', markersize=10),
        ]
    plt.legend(handles=legend_elements_prototypes, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'PCA of Prototypes - Round {round_num}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

    save_path = os.path.join(save_dir, f'pca_prototypes_round_{round_num}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Prototypes PCA plot saved to {save_path}")

    # Log the second plot to wandb
    wandb.log({
        f"PCA of Prototypes - Round {round_num}": wandb.Image(save_path),
        "round": round_num
    })