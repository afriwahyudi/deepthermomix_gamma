# datasplit_module.py

from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
import copy

from typing import List, Tuple, Union
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def system_disjoint_split(
    data: Union[List[Data], pd.DataFrame],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by_components: bool = False
) -> Tuple:
    """
    Split PyG Data graphs OR a Pandas DataFrame into disjoint train/val/test sets by system_id.
    Args:
        data (List[Data] | pd.DataFrame): Either a list of PyG Data objects or a Pandas DataFrame.
                                          Both must have a 'system_id' attribute/column.
        test_size (float): Fraction of total data to use as test set
        val_size (float): Fraction of remaining data to use as validation set
        random_state (int): Random seed for reproducibility
        stratify_by_components (bool): If True, stratify splits to maintain proportion of 
                                       binary/ternary/n-ary mixtures in each split.
    Returns:
        train_data, val_data, test_data
        (same type as input)
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        if 'system_id' not in data.columns:
            raise ValueError("DataFrame must contain a 'system_id' column.")
        
        # Get unique system_id and their component counts
        if stratify_by_components:
            if 'component_names' not in data.columns and 'component_list' not in data.columns:
                raise ValueError("DataFrame must contain 'component_names' or 'component_list' for stratification.")
            
            comp_col = 'component_names' if 'component_names' in data.columns else 'component_list'
            system_info = data.groupby('system_id')[comp_col].first().apply(len)
            system_ids = system_info.index.values
            component_counts = system_info.values
        else:
            system_ids = data['system_id'].unique()
            component_counts = None
    elif isinstance(data, list):
        # Build system_id -> component_count mapping
        system_component_map = {}
        for g in data:
            if g.system_id not in system_component_map:
                system_component_map[g.system_id] = len(g.component_names)
        
        system_ids = np.array(list(system_component_map.keys()))
        component_counts = np.array([system_component_map[sid] for sid in system_ids]) if stratify_by_components else None
    else:
        raise TypeError("Input must be a list of PyG Data objects or a pandas DataFrame.")
    
    # Split system IDs into train/val/test with optional stratification
    if stratify_by_components:
        train_val_ids, test_ids, train_val_counts, test_counts = train_test_split(
            system_ids, component_counts,
            test_size=test_size, 
            random_state=random_state,
            stratify=component_counts
        )
        relative_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=relative_val_size, 
            random_state=random_state,
            stratify=train_val_counts
        )
    else:
        train_val_ids, test_ids = train_test_split(
            system_ids, test_size=test_size, random_state=random_state
        )
        relative_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=relative_val_size, random_state=random_state
        )
    
    # Convert to sets for O(1) lookup
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    test_ids_set = set(test_ids)
    
    if is_dataframe:
        # Use boolean indexing directly (fastest for DataFrames)
        system_id_col = data['system_id']
        train_data = data[system_id_col.isin(train_ids_set)].copy()
        val_data = data[system_id_col.isin(val_ids_set)].copy()
        test_data = data[system_id_col.isin(test_ids_set)].copy()
    else:
        # Pre-allocate lists
        train_data = []
        val_data = []
        test_data = []
        
        # Single pass through data
        for g in data:
            if g.system_id in train_ids_set:
                train_data.append(g)
            elif g.system_id in val_ids_set:
                val_data.append(g)
            else:  # must be in test_ids_set
                test_data.append(g)
    
    # Print component distribution
    if stratify_by_components:
        if is_dataframe:
            comp_col = 'component_names' if 'component_names' in data.columns else 'component_list'
            train_comp_dist = train_data[comp_col].apply(len).value_counts().sort_index()
            val_comp_dist = val_data[comp_col].apply(len).value_counts().sort_index()
            test_comp_dist = test_data[comp_col].apply(len).value_counts().sort_index()
        else:
            from collections import Counter
            train_comp_dist = Counter(len(g.component_names) for g in train_data)
            val_comp_dist = Counter(len(g.component_names) for g in val_data)
            test_comp_dist = Counter(len(g.component_names) for g in test_data)
            # Convert to sorted dict for consistent display
            train_comp_dist = dict(sorted(train_comp_dist.items()))
            val_comp_dist = dict(sorted(val_comp_dist.items()))
            test_comp_dist = dict(sorted(test_comp_dist.items()))
        
        print("Component distribution:")
        print(f"  Train: {dict(train_comp_dist)}")
        print(f"  Val:   {dict(val_comp_dist)}")
        print(f"  Test:  {dict(test_comp_dist)}")
    
    print(f"\nDatapoints -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Unique systems -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_data, val_data, test_data