import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

EPOCHS = 80
N_TRIALS = 100
BATCH_SIZE = 1024
K_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001

def load_train_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

def load_all_graphs_and_features(train_df):
    unique_keys = train_df[["group", "traj", "frame"]].drop_duplicates()
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
        y_tensor = torch.full((x_tensor.size(0), ), -1, dtype=torch.long)
        data_pyg = Data(x=x_tensor, edge_index=edge_index_tensor)
        data_pyg.y = y_tensor
        data_pyg = data_pyg.to(DEVICE)
        graphs_dict[(g, t, f)] = data_pyg
    for i, row in train_df.iterrows():
        g, t, f = int(row["group"]), int(row["traj"]), int(row["frame"])
        node_idx = int(row["particle"])
        label = int(row["label"])
        graph_data = graphs_dict[(g, t, f)]
        graph_data.y[node_idx] = label
        samples.append(((g, t, f), node_idx, label))
    return graphs_dict, samples

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

def train_one_epoch(model_gnn, model_mlp, optimizer, graphs_dict, samples):
    model_gnn.train()
    model_mlp.train()
    np.random.shuffle(samples)
    total_loss = 0.0
    num_batches = int(np.ceil(len(samples) / BATCH_SIZE))
    idx_start = 0
    for _ in range(num_batches):
        batch_samples = samples[idx_start: idx_start + BATCH_SIZE]
        idx_start += BATCH_SIZE
        optimizer.zero_grad()
        batch_loss = 0.0
        grouped_by_graph = {}
        for ((g, t, f), node_idx, label) in batch_samples:
            if (g, t, f) not in grouped_by_graph:
                grouped_by_graph[(g, t, f)] = []
            grouped_by_graph[(g, t, f)].append((node_idx, label))
        for k, node_label_list in grouped_by_graph.items():
            graph_data = graphs_dict[k]
            x, edge_index = graph_data.x, graph_data.edge_index
            node_embeddings = model_gnn(x, edge_index)
            node_indices = [nl[0] for nl in node_label_list]
            labels = [nl[1] for nl in node_label_list]
            labels_t = torch.tensor(labels, dtype=torch.long, device=DEVICE)
            selected_emb = node_embeddings[node_indices]
            logits = model_mlp(selected_emb)
            loss = F.cross_entropy(logits, labels_t)
            batch_loss += loss
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model_gnn, model_mlp, graphs_dict, samples):
    model_gnn.eval()
    model_mlp.eval()
    preds = []
    trues = []
    for (g, t, f), node_idx, label in samples:
        graph_data = graphs_dict[(g, t, f)]
        x, edge_index = graph_data.x, graph_data.edge_index
        node_embeddings = model_gnn(x, edge_index)
        logit = model_mlp(node_embeddings[node_idx].unsqueeze(0))
        prob = F.softmax(logit, dim=-1)[0]
        pred_label = prob.argmax().item()
        preds.append(pred_label)
        trues.append(label)
    preds_proba = []
    for (g, t, f), node_idx, label in samples:
        graph_data = graphs_dict[(g, t, f)]
        x, edge_index = graph_data.x, graph_data.edge_index
        node_embeddings = model_gnn(x, edge_index)
        logit = model_mlp(node_embeddings[node_idx].unsqueeze(0))
        prob = F.softmax(logit, dim=-1)[0]
        preds_proba.append(prob[1].item())
    y_true = np.array(trues)
    y_score = np.array(preds_proba)
    if len(np.unique(y_true)) == 2:
        auc_val = roc_auc_score(y_true, y_score)
    else:
        auc_val = 0.5
    acc_val = accuracy_score(trues, preds)
    return auc_val, acc_val

def train_with_early_stopping(
    model_gnn,
    model_mlp,
    optimizer,
    graphs_dict,
    train_samples,
    val_samples,
    max_epochs,
    patience,
    min_delta=0.0,
    pbar=None,
    trial_number=None,
    fold_index=None,
    total_folds=None
):
    best_val_auc = -float("inf")
    no_improve_count = 0
    best_gnn_state = None
    best_mlp_state = None
    epochs_done = 0

    for ep in range(max_epochs):
        train_one_epoch(model_gnn, model_mlp, optimizer, graphs_dict, train_samples)
        val_auc, _ = evaluate(model_gnn, model_mlp, graphs_dict, val_samples)
        if pbar is not None:
            tn = trial_number + 1 if trial_number is not None else "?"
            fi = fold_index + 1 if fold_index is not None else "?"
            tf = total_folds if total_folds is not None else "?"
            pbar.set_description(
                f"Trial {tn}, epoch {ep+1}/{max_epochs}, fold {fi}/{tf} (AUC={val_auc:.4f})"
            )
            pbar.update(1)
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            no_improve_count = 0
            best_gnn_state = {k: v.cpu() for k, v in model_gnn.state_dict().items()}
            best_mlp_state = {k: v.cpu() for k, v in model_mlp.state_dict().items()}
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            break
        epochs_done = ep + 1

    if best_gnn_state is not None:
        model_gnn.load_state_dict(best_gnn_state)
        model_mlp.load_state_dict(best_mlp_state)
    return best_val_auc, epochs_done

def cross_validation_score(trial, graphs_dict, samples):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    all_auc = []
    total_steps = K_FOLDS * EPOCHS
    with tqdm(total=total_steps, leave=False) as pbar:
        for fold_i, (train_index, val_index) in enumerate(kf.split(samples)):
            train_samples = samples[train_index]
            val_samples = samples[val_index]
            hparams = trial.user_attrs["current_hparams"]
            model_gnn = GraphSAGE(
                in_channels=10,
                hidden_channels=hparams["gnn_hidden_dim"],
                num_layers=hparams["gnn_num_layers"],
                dropout=hparams["dropout"],
            ).to(DEVICE)
            model_mlp = ClassifierMLP(
                in_dim=hparams["gnn_hidden_dim"],
                hidden_dim=hparams["mlp_hidden_dim"],
                num_layers=hparams["mlp_num_layers"],
                dropout=hparams["dropout"],
            ).to(DEVICE)
            optimizer = torch.optim.Adam(
                list(model_gnn.parameters()) + list(model_mlp.parameters()),
                lr=hparams["lr"]
            )
            best_val_auc, _ = train_with_early_stopping(
                model_gnn=model_gnn,
                model_mlp=model_mlp,
                optimizer=optimizer,
                graphs_dict=graphs_dict,
                train_samples=train_samples,
                val_samples=val_samples,
                max_epochs=EPOCHS,
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=MIN_DELTA,
                pbar=pbar,
                trial_number=trial.number,
                fold_index=fold_i,
                total_folds=K_FOLDS
            )
            all_auc.append(best_val_auc)
    return float(np.mean(all_auc))

def objective(trial, graphs_dict, samples):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    gnn_num_layers = trial.suggest_int("gnn_num_layers", 1, 3)
    gnn_hidden_dim = trial.suggest_int("gnn_hidden_dim", 16, 128)
    mlp_num_layers = trial.suggest_int("mlp_num_layers", 1, 3)
    mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hparams = {
        "lr": lr,
        "gnn_num_layers": gnn_num_layers,
        "gnn_hidden_dim": gnn_hidden_dim,
        "mlp_num_layers": mlp_num_layers,
        "mlp_hidden_dim": mlp_hidden_dim,
        "dropout": dropout
    }
    trial.set_user_attr("current_hparams", hparams)
    avg_auc = cross_validation_score(trial, graphs_dict, np.array(samples, dtype=object))
    return avg_auc

def main():
    train_csv = "data/dataset_index/Task_4_train.csv"
    train_df = load_train_data(train_csv)
    graphs_dict, samples = load_all_graphs_and_features(train_df)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, graphs_dict, samples), n_trials=N_TRIALS)
    print("Best trial:")
    print(study.best_trial)
    best_hparams = study.best_trial.params
    print("Best hyperparameters:", best_hparams)

    final_model_gnn = GraphSAGE(
        in_channels=10,
        hidden_channels=best_hparams["gnn_hidden_dim"],
        num_layers=best_hparams["gnn_num_layers"],
        dropout=best_hparams["dropout"],
    ).to(DEVICE)
    final_model_mlp = ClassifierMLP(
        in_dim=best_hparams["gnn_hidden_dim"],
        hidden_dim=best_hparams["mlp_hidden_dim"],
        num_layers=best_hparams["mlp_num_layers"],
        dropout=best_hparams["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(final_model_gnn.parameters()) + list(final_model_mlp.parameters()),
        lr=best_hparams["lr"]
    )
    print("\n=== Now training the final model with the best hyperparameters (no early stopping) ===")
    for ep in range(EPOCHS):
        loss_val = train_one_epoch(final_model_gnn, final_model_mlp, optimizer, graphs_dict, samples)
        print(f"Final Training Epoch {ep+1}/{EPOCHS}, Loss: {loss_val:.4f}")
    os.makedirs("saved_model", exist_ok=True)
    torch.save({
        "gnn_state_dict": final_model_gnn.state_dict(),
        "mlp_state_dict": final_model_mlp.state_dict(),
    }, os.path.join("saved_model", "model_4.pth"))
    with open(os.path.join("saved_model", "hyper_4.json"), "w") as f:
        json.dump(best_hparams, f, indent=2)
    print("=== Training complete. Model (model_4.pth) and hyperparameters (hyper_4.json) have been saved. ===")

if __name__ == "__main__":
    main()
