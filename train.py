import argparse
import numpy as np
import scipy.sparse as sp
from utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import Encoder_Net
from loader import load_dataset

# Graph
# python train.py --dataset cora --cluster_num 7
# python train.py --dataset citeseer --cluster_num 6
# python train.py --dataset amap --cluster_num 8
# python train.py --dataset pubmed --cluster_num 3

# Hapergraph
# python train.py --dataset NTU2012
# python train.py --dataset zoo --cluster_num 7
# python train.py --dataset ModelNet40 --cluster_num 40
# python train.py --dataset 20newsW100 --cluster_num 4

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=4, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=500, help='feature dim')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='NTU2012', help='name of dataset.')
parser.add_argument('--cluster_num', type=int, default=67, help='number of cluster.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--threshold', type=float, default=0.1, help='the threshold of high-confidence')
parser.add_argument('--alpha', type=float, default=0.5, help='trade-off of loss')
parser.add_argument('--tau1', type=float, default=0.5, help='InfoNCE temperature parameter')
parser.add_argument('--tau2', type=float, default=0.5, help='InfoNCE temperature parameter')
args = parser.parse_args()

# Device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.device = device

# Load Data
features, true_labels, adj = load_dataset(args.dataset)
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)

# Smoothing
adj_norm_s = preprocess_graph(adj, args.t, norm='sym', renorm=True)
smooth_fea = sp.csr_matrix(features).toarray()
for a in adj_norm_s:
    smooth_fea = a.dot(smooth_fea)
smooth_fea = torch.FloatTensor(smooth_fea)

# Loss Function
def infonce_loss(anchor, positive, negatives, temperature):

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # [batch_size]
    
    neg_sim = anchor @ negatives.T / temperature  # [batch_size, num_negatives]
    
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+num_negatives]
    
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss

def cluster_level_infonce(cluster_centers1, cluster_centers2, temperature):

    num_clusters = cluster_centers1.shape[0]
    total_loss = 0.0
    
    for c in range(num_clusters):

        anchor = cluster_centers1[c:c+1]  # [1, dim]

        positive = cluster_centers2[c:c+1]  # [1, dim]

        negatives = torch.cat([cluster_centers2[:c], cluster_centers2[c+1:]])  # [num_clusters-1, dim]
        
        if len(negatives) == 0:
            continue

        loss = infonce_loss(anchor, positive, negatives, temperature)
        total_loss += loss
    
    return total_loss / num_clusters if num_clusters > 0 else 0.0

def sample_level_infonce(z1, z2, cluster_centers, predict_labels, temperature):

    batch_size = z1.shape[0]
    num_clusters = cluster_centers.shape[0]
    total_loss = 0.0
    
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cluster_centers = F.normalize(cluster_centers, dim=-1)
    
    for i in range(batch_size):

        anchor = z1[i:i+1]  # [1, dim]

        positive = z2[i:i+1]  # [1, dim]
        

        curr_cluster = predict_labels[i]

        negatives = []
        for c in range(num_clusters):
            if c != curr_cluster:
                negatives.append(cluster_centers[c:c+1])
        
        if len(negatives) == 0:
            continue
            
        negatives = torch.cat(negatives, dim=0)  # [num_neg_clusters, dim]
        
        loss = infonce_loss(anchor, positive, negatives, temperature)
        total_loss += loss
    
    return total_loss / batch_size if batch_size > 0 else 0.0

def batch_sample_level_infonce(z1, z2, cluster_centers, predict_labels, temperature):

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cluster_centers = F.normalize(cluster_centers, dim=-1)
    
    batch_size = z1.shape[0]
    num_clusters = cluster_centers.shape[0]
    
    pos_sim = (z1 * z2).sum(dim=-1) / temperature  # [batch_size]
    
    cluster_sim = z1 @ cluster_centers.T / temperature  # [batch_size, num_clusters]
    
    neg_sim_list = []
    for i in range(batch_size):
        curr_cluster = predict_labels[i]

        neg_sim = torch.cat([
            cluster_sim[i, :curr_cluster],
            cluster_sim[i, curr_cluster+1:]
        ])
        neg_sim_list.append(neg_sim)
    

    max_neg_num = max([len(neg) for neg in neg_sim_list] + [0])
    padded_neg_sim = torch.zeros(batch_size, max_neg_num, device=z1.device)
    mask = torch.zeros(batch_size, max_neg_num, device=z1.device, dtype=torch.bool)
    
    for i, neg_sim in enumerate(neg_sim_list):
        if len(neg_sim) > 0:
            padded_neg_sim[i, :len(neg_sim)] = neg_sim
            mask[i, :len(neg_sim)] = True
    

    pos_sim = pos_sim.unsqueeze(1)  # [batch_size, 1]
    logits = torch.cat([pos_sim, padded_neg_sim], dim=1)  # [batch_size, 1+max_neg_num]
    

    labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)
    loss = F.cross_entropy(logits, labels, reduction='none')
    
 
    has_neg = mask.any(dim=1)
    if has_neg.sum() > 0:
        loss = loss[has_neg].mean()
    else:
        loss = torch.tensor(0.0, device=z1.device)
    
    return loss

# Train

print(f"===== Train")

acc_list = []
nmi_list = []
ari_list = []
f1_list = []

for run in range(5):

    # init
    setup_seed(run)
    best_acc, best_nmi, best_ari, best_f1, predict_labels, dis = clustering(smooth_fea, true_labels, args.cluster_num)
    n_nodes = smooth_fea.shape[0]

    # MLP
    model = Encoder_Net(args.linlayers, [features.shape[1]] + [args.dims])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # GPU
    model.to(device)
    smooth_fea = smooth_fea.to(device)
    
    for epoch in tqdm(range(args.epochs), desc=f"Run {run}"):
        
        model.train()
        optimizer.zero_grad()  
        
        z1, z2 = model(smooth_fea, smooth_fea)

        cluster_loss = torch.tensor(0.0, device=device)
        sample_loss = torch.tensor(0.0, device=device)

        # 1. Cluster-level contrastive learning
        if epoch >= 0:
            predict_labels_tensor = torch.tensor(predict_labels, device=device)
            cluster_centers_z1 = torch.zeros(args.cluster_num, args.dims, device=device)
            cluster_centers_z2 = torch.zeros(args.cluster_num, args.dims, device=device)
            
            for c in range(args.cluster_num):
                cluster_mask = (predict_labels_tensor == c)
                if cluster_mask.sum() == 0:
                    cluster_centers_z1[c] = torch.randn(args.dims, device=device)
                    cluster_centers_z2[c] = torch.randn(args.dims, device=device)
                    continue
                cluster_centers_z1[c] = z1[cluster_mask].mean(dim=0)
                cluster_centers_z2[c] = z2[cluster_mask].mean(dim=0)
            
            cluster_loss = cluster_level_infonce(
                cluster_centers_z1, 
                cluster_centers_z2, 
                args.tau2
            )

        # 2. Sample-level contrastive learning
        if epoch >= 0:
            # 复用簇中心
            cluster_centers = (cluster_centers_z1 + cluster_centers_z2) / 2
            
            # 计算样本级InfoNCE损失（批量版本更高效）
            sample_loss = batch_sample_level_infonce(
                z1, z2, cluster_centers, 
                predict_labels, args.tau1
            )

        loss = (1 - args.alpha) * cluster_loss + args.alpha * sample_loss

        loss.backward()
        optimizer.step()

        # 0. Rough Clustering (every 10 Epochs)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z1_eval, z2_eval = model(smooth_fea, smooth_fea)
                hidden_emb = (z1_eval + z2_eval) / 2

                acc, nmi, ari, f1, predict_labels, dis = clustering(
                    torch.cat([z1_eval, z2_eval], dim=1).cpu(), 
                    true_labels, 
                    args.cluster_num
                )
                
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1

    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)
    print(f"Run {run} - Epoch: {epoch}, Acc: {best_acc:.4f}, NMI: {best_nmi:.4f}, ARI: {best_ari:.4f}, F1: {best_f1:.4f}")

# Final Results

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)

print(f"===== Final Results")
print(f"ACC: {acc_list.mean():.4f} ± {acc_list.std():.4f}")
print(f"NMI: {nmi_list.mean():.4f} ± {nmi_list.std():.4f}")
print(f"ARI: {ari_list.mean():.4f} ± {ari_list.std():.4f}")
print(f"F1: {f1_list.mean():.4f} ± {f1_list.std():.4f}")