import torch
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.data import DataLoader
import os

# --- MODELLO ---
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(Seq(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))
        self.conv2 = GINConv(Seq(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)

# --- UTILITY ---
def load_graphs(folder):
    return [torch.load(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".pt")]

def compute_embeddings(model, loader, device):
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch.x, batch.edge_index, batch.batch)
            embs.append(emb.cpu())
    return torch.cat(embs, dim=0)

# --- MAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GIN(in_channels=512, hidden_channels=256, out_channels=128).to(device)

bari_graphs = load_graphs("data/gnn_ready_graphs/Bari")
shibuya_graphs = load_graphs("data/gnn_ready_graphs/Shibuya")

bari_loader = DataLoader(bari_graphs, batch_size=1)
shibuya_loader = DataLoader(shibuya_graphs, batch_size=1)

bari_emb = compute_embeddings(model, bari_loader, device)
shibuya_emb = compute_embeddings(model, shibuya_loader, device)

# --- RETRIEVAL ---
import torch.nn.functional as F

similarities = F.cosine_similarity(bari_emb.unsqueeze(1), shibuya_emb.unsqueeze(0), dim=-1)  # [n_bari, n_shibuya]
top_match = similarities.argmax(dim=1)  # best shibuya image for each bari
topk = similarities.topk(5, dim=1).indices  # top-5 most similar shibuya images for each bari

# Salvataggio pairing
import pandas as pd
df = pd.DataFrame({
    "bari_id": list(range(len(bari_graphs))),
    "best_shibuya_id": top_match.tolist(),
    "top5_shibuya_ids": topk.tolist()
})
df.to_csv("bari_to_shibuya_retrieval.csv", index=False)
print("✅ Retrieval completato. File salvato in 'bari_to_shibuya_retrieval.csv'")
