import torch
import torch.nn as nn
from modules.dtmpnn import DTMPNN

def infer_model_architecture(state_dict, checkpoint=None):
    """
    Infer DTMPNN architecture parameters from state_dict keys and shapes.
    """
    node_dim = state_dict['graph_block.layers.0.lin_node.weight'].shape[1]
    edge_dim = state_dict['graph_block.layers.0.lin_edge.weight'].shape[1]
    graph_hidden_dim = state_dict['graph_block.layers.0.lin_node.weight'].shape[0]
    latent_dim = state_dict['mixture_layer.gate_mlp.0.weight'].shape[0]
    context_dim = state_dict['mixture_layer.interaction_mlp.0.weight'].shape[0]
    fc_hidden_dim = state_dict['fc_block.layers.0.weight'].shape[0]
    fc_layer_indices = [int(k.split('.')[2]) for k in state_dict.keys() 
                        if k.startswith('fc_block.layers.') and k.split('.')[2].isdigit() and 'weight' in k]
    last_fc_layer = max(fc_layer_indices)
    output_dim = state_dict[f'fc_block.layers.{last_fc_layer}.weight'].shape[0]
    layer_indices = [int(k.split('.')[2]) for k in state_dict.keys() 
                     if k.startswith('graph_block.layers.') and k.split('.')[2].isdigit()]
    graph_layers = max(layer_indices) + 1 if layer_indices else 1
    fc_layers = len([k for k in state_dict.keys() if k.startswith('fc_block.layers.') and 'weight' in k])
    fc_input_dim = state_dict['fc_block.layers.0.weight'].shape[1]
    num_global_features = fc_input_dim - latent_dim
  
    return {
        'node_dim': node_dim,
        'edge_dim': edge_dim,
        'graph_hidden_dim': graph_hidden_dim,
        'latent_dim': latent_dim,
        'context_dim': context_dim,
        'fc_hidden_dim': fc_hidden_dim,
        'output_dim': output_dim,
        'num_global_features': num_global_features,
        'graph_layers': graph_layers,
        'fc_layers': fc_layers
    }

def load_model(checkpoint_path, return_stats=False, verbose=True):
    """
    Load DTMPNN model with automatic architecture inference.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        return_stats: If True, returns (model, stats). If False, returns just the model
        verbose: If True, prints loading information
    
    Returns:
        model: Loaded DTMPNN model (in eval mode)
        stats: Training stats dict (only if return_stats=True)
    
    Example:
        model = load_model('checkpoints/00_dummy.pt')
        model, stats = load_model('checkpoints/00_dummy.pt', return_stats=True)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            if 'graph_block.layers.0.lin_node.weight' in checkpoint:
                state_dict = checkpoint
            else:
                for v in checkpoint.values():
                    if isinstance(v, dict) and any('weight' in k for k in v.keys()):
                        state_dict = v
                        break
                else:
                    state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model_params = infer_model_architecture(state_dict, checkpoint)
    
    if verbose:
        print("Inferred model parameters:")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
    
    model = DTMPNN(**model_params)
    model.load_state_dict(state_dict)
    model.eval()
    
    if verbose:
        print("\nModel loaded successfully!")
    if return_stats:
        stats = checkpoint.get('stats', None) if isinstance(checkpoint, dict) else None
        if verbose and stats:
            print("Stats available in checkpoint")
        return model, stats
    
    return model