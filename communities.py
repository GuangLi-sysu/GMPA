import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def graphsage_cl(x, edge_index, embed_channel=32, iter_num=500, model_file=None, device=torch.device('cpu'), model_name=None):
    model = GraphSAGE(x.size(1), embed_channel)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index) 

        pos_edge_index = edge_index
        neg_edge_index = torch.randint(0, x.size(0), pos_edge_index.size(), dtype=torch.long)

        pos_loss = -torch.log(
            F.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))+1e-9).mean()
        neg_loss = -torch.log(
            F.sigmoid(-(z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))+1e-9).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        return loss

    if model_file:
        model.load_state_dict(torch.load(model_file))
        model.eval()
        with torch.no_grad():
            embeds = model(x, edge_index)
        return embeds.cpu()
    else:
        for epoch in range(iter_num):
            loss = train()

        model.eval()
        with torch.no_grad():
            embeds = model(x, edge_index)
        return embeds.cpu(), model