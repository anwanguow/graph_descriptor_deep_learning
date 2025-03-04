import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class ClassifierMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.linears = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.linears.append(nn.Linear(in_dim, 2))
        else:
            self.linears.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, 2))
    def forward(self, x):
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = self.linears[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.linears[i](x)
        return x

def load_test_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

def load_all_test_graphs_and_features(test_df):
    unique_keys = test_df[["group", "traj", "frame"]].drop_duplicates()
    graphs_dict = {}
    samples = []
    for idx, row in unique_keys.iterrows():
        g, t, f = int(row["group"]), int(row["traj"]), int(row["frame"])
        adj_path = f"data/graph/set_{g}/D_{t}/T_{f}_r_2.5_f_0.55.npy"
        x_path = f"data/X/set_{g}/D_{t}/X_{f}_r_2.5_f_0.55.npy"
        adj_matrix = np.load(adj_path)
        x_matrix = np.load(x_path)
        edge_index_np = np.vstack(np.nonzero(adj_matrix))
        edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
        x_tensor = torch.tensor(x_matrix, dtype=torch.float)
        data_pyg = Data(x=x_tensor, edge_index=edge_index_tensor)
        data_pyg = data_pyg.to(DEVICE)
        graphs_dict[(g, t, f)] = data_pyg
    for i, row in test_df.iterrows():
        g, t, f = int(row["group"]), int(row["traj"]), int(row["frame"])
        node_idx = int(row["particle"])
        label = int(row["label"])
        samples.append(((g, t, f), node_idx, label))
    return graphs_dict, samples

def main():
    checkpoint = torch.load("saved_model/model_1.pth", map_location=DEVICE)
    with open("saved_model/hyper_1.json", "r") as f:
        best_hparams = json.load(f)
    model_gnn = GraphSAGE(
        in_channels=10,
        hidden_channels=best_hparams["gnn_hidden_dim"],
        num_layers=best_hparams["gnn_num_layers"],
        dropout=best_hparams["dropout"],
    ).to(DEVICE)
    model_mlp = ClassifierMLP(
        in_dim=best_hparams["gnn_hidden_dim"],
        hidden_dim=best_hparams["mlp_hidden_dim"],
        num_layers=best_hparams["mlp_num_layers"],
        dropout=best_hparams["dropout"],
    ).to(DEVICE)
    model_gnn.load_state_dict(checkpoint["gnn_state_dict"])
    model_mlp.load_state_dict(checkpoint["mlp_state_dict"])
    model_gnn.eval()
    model_mlp.eval()
    test_csv = "data/dataset_index/Task_1_test.csv"
    test_df = load_test_data(test_csv)
    graphs_dict, test_samples = load_all_test_graphs_and_features(test_df)
    preds = []
    preds_proba = []
    trues = []
    for (g, t, f), node_idx, label in test_samples:
        data_pyg = graphs_dict[(g, t, f)]
        x, edge_index = data_pyg.x, data_pyg.edge_index
        with torch.no_grad():
            node_emb = model_gnn(x, edge_index)
            logit = model_mlp(node_emb[node_idx].unsqueeze(0))
            prob = F.softmax(logit, dim=-1)[0]
            pred_label = prob.argmax().item()
        preds.append(pred_label)
        preds_proba.append(prob[1].item())
        trues.append(label)
    cm = confusion_matrix(trues, preds)
    mcc_val = matthews_corrcoef(trues, preds)
    auc_val = roc_auc_score(trues, preds_proba) if len(set(trues)) > 1 else 0.5
    acc_val = accuracy_score(trues, preds)
    result_str = (
        f"Confusion Matrix:\n{cm}\n"
        f"MCC: {mcc_val:.4f}\n"
        f"AUC: {auc_val:.4f}\n"
        f"Accuracy: {acc_val:.4f}\n"
    )
    print(result_str)
    os.makedirs("results", exist_ok=True)
    with open("results/task_1.txt", "w") as f:
        f.write(result_str)
    print("=== 测试完成，结果已保存到 results/task_1.txt ===")

if __name__ == "__main__":
    main()