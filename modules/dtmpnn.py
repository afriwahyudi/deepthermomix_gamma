import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops

# 1. Custom MPNN Layer
class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, graph_hidden_dim):
        super(MPNNLayer, self).__init__(aggr="add")
        self.lin_node = nn.Linear(node_dim, graph_hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, graph_hidden_dim)

        self.message_mlp = nn.Sequential(
            nn.Linear(graph_hidden_dim + graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(graph_hidden_dim + graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )

    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x_orig):
        update_input = torch.cat([aggr_out, x_orig], dim=-1)
        return self.update_mlp(update_input)
        
    def forward(self, x, edge_index, edge_attr, mol_batch):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=0)
        x_transformed = self.lin_node(x)
        edge_attr_transformed = self.lin_edge(edge_attr)
        node_emb = self.propagate(edge_index, x=x_transformed, 
                                  edge_attr=edge_attr_transformed, x_orig=x_transformed)

        return node_emb

# 2. Graph Neural Network Block
class MPNNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, graph_hidden_dim, num_layers=3):
        super(MPNNBlock, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(MPNNLayer(node_dim, edge_dim, graph_hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(MPNNLayer(graph_hidden_dim, edge_dim, graph_hidden_dim))
        if num_layers > 1:
            self.skip_weights = nn.Parameter(torch.ones(num_layers - 1))

    def forward(self, x, edge_index, edge_attr, mol_batch):
        node_emb          = self.layers[0](x, edge_index, edge_attr, mol_batch)
        for i in range(1, self.num_layers):
            node_emb_new  = self.layers[i](node_emb, edge_index, edge_attr, mol_batch)
            node_emb      = node_emb_new + torch.relu(self.skip_weights[i-1]) * node_emb
        comp_emb = global_mean_pool(node_emb, mol_batch)
                
        return node_emb, comp_emb
    
# 3. DeepThermoMix
class DeepThermoMix(nn.Module):
    def __init__(self, component_emb_dim, latent_dim, context_dim):
        super(DeepThermoMix, self).__init__()
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(component_emb_dim + 1, context_dim),
            nn.Softplus(),          
            nn.Linear(context_dim, context_dim),
            nn.Softplus()
        )
        
        gate_input_dim = component_emb_dim + 1 + context_dim
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, latent_dim),
            nn.Softplus(),
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus()
        )

    def forward(self, comp_emb, mole_frac,
                component_batch_batch,
                track_grad=False): 
        
        if track_grad:
            mole_frac = mole_frac.requires_grad_(True)
        
        mole_frac_view = mole_frac.view(-1, 1)

        raw_context_input = torch.cat([comp_emb, mole_frac_view], dim=-1)
        mixture_context_pooled = global_mean_pool(raw_context_input, component_batch_batch)
        mixture_context_vec = self.interaction_mlp(mixture_context_pooled)
        mixture_context_expanded = mixture_context_vec[component_batch_batch]

        gate_input = torch.cat([
            comp_emb, 
            mole_frac_view, 
            mixture_context_expanded
        ], dim=-1)
        
        latent_vectors = self.gate_mlp(gate_input)

        return latent_vectors, mole_frac

# 4. Parameterizable Full Model
class DTMPNN(nn.Module):
    def __init__(self, 
                 node_dim, 
                 edge_dim, 
                 graph_hidden_dim, 
                 latent_dim, 
                 context_dim,
                 graph_layers=3,
                 track_grad=False):
        """
        Args:
            node_dim           : Dimensionality of node features
            edge_dim           : Dimensionality of edge features
            graph_hidden_dim   : Hidden dimension for MPNN
            graph_layers       : Number of MPNN layers
            latent_dim         : Hidden dimension for DeepThermoMix
        """
        super(DTMPNN, self).__init__()
        self.track_grad    = track_grad
        self.graph_block   = MPNNBlock(node_dim, edge_dim, graph_hidden_dim, num_layers=graph_layers)
        self.mixture_layer = DeepThermoMix(graph_hidden_dim, latent_dim, context_dim)
        self.output_layer  = nn.Linear(latent_dim, 1)

    def forward(self, data):
        """
        Args:
            data: PyG Data object with attributes:
                  - x, edge_index, edge_attr, mol_batch
                  - component_batch
                  - component_mole_frac
        """
        node_emb, comp_emb = self.graph_block(data.x, data.edge_index, 
                                              data.edge_attr, data.mol_batch)
        latent_vectors, mole_frac = self.mixture_layer(
            comp_emb, 
            data.component_mole_frac,
            data.component_batch_batch,
            self.track_grad)
        
        ln_gamma_calc = self.output_layer(latent_vectors).squeeze(-1)

        return ln_gamma_calc, latent_vectors, comp_emb 