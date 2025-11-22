import torch
import torch.nn as nn
import numpy as np 
from modules.dtmpnn import DTMPNN 

def infer_model_architecture(state_dict):
    """
    Infer DTMPNN architecture parameters from state_dict keys and shapes
    based on the specific DeepThermoMix architecture.
    """
    # 1. Infer Graph Dimensions
    node_dim = state_dict['graph_block.layers.0.lin_node.weight'].shape[1]
    edge_dim = state_dict['graph_block.layers.0.lin_edge.weight'].shape[1]
    graph_hidden_dim = state_dict['graph_block.layers.0.lin_node.weight'].shape[0]

    # 2. Infer Mixture/Interaction Dimensions
    context_dim = state_dict['mixture_layer.interaction_mlp.0.weight'].shape[0]
    latent_dim = state_dict['mixture_layer.gate_mlp.0.weight'].shape[0]

    # 3. Infer Depth
    layer_indices = [int(k.split('.')[2]) for k in state_dict.keys() 
                     if k.startswith('graph_block.layers.') and k.split('.')[2].isdigit()]
    graph_layers = max(layer_indices) + 1 if layer_indices else 1

    return {
        'node_dim': node_dim,
        'edge_dim': edge_dim,
        'graph_hidden_dim': graph_hidden_dim,
        'latent_dim': latent_dim,
        'context_dim': context_dim,
        'graph_layers': graph_layers,
    }

def load_model(checkpoint_path, constraint_type='soft', verbose=True):
    """
    Load DTMPNN model with automatic architecture inference.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file.
        constraint_type: 'soft' or 'hard'. Defaults to 'soft'.
        verbose: If True, prints loading information.
    
    Returns:
        model: Loaded DTMPNN model (in eval mode).
    """
    # --- 1. Register Safe Globals ---
    try:
        # Try to whitelist numpy scalar if available
        if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
             torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass # specific numpy version issues can be ignored if we use weights_only=False

    # --- 2. Load Checkpoint ---
    try:
        # Try loading with security relaxation (PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions lacking 'weights_only'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # --- 3. Handle Checkpoint Structure ---
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Heuristic check
            if 'graph_block.layers.0.lin_node.weight' in checkpoint:
                state_dict = checkpoint
            else:
                state_dict = checkpoint 
    else:
        state_dict = checkpoint
    
    # --- 4. Infer Architecture ---
    model_params = infer_model_architecture(state_dict)
    
    # Set constraint type (manual override)
    model_params['constraint_type'] = constraint_type
    
    # Check if config was saved in checkpoint (auto override)
    if isinstance(checkpoint, dict):
        saved_args = checkpoint.get('args', {}) or checkpoint.get('config', {})
        if isinstance(saved_args, dict) and 'constraint_type' in saved_args:
            model_params['constraint_type'] = saved_args['constraint_type']
            if verbose: 
                print(f"Found constraint_type='{model_params['constraint_type']}' in checkpoint config.")

    if verbose:
        print("Inferred model parameters:")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
    
    # --- 5. Initialize & Load Weights ---
    model = DTMPNN(**model_params)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: strict loading failed ({e}). Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    
    if verbose:
        print("\nModel loaded successfully!")

    return model