import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import torch
from torch_geometric.data import Data
from typing import List, Tuple
import re

class DataPipeline:
    """
    High-level pipeline for processing N-multicomponent mixture
    into canonicalized form and model-ready representations.
    Two necessary files to be supplied:
        1. Component registry CSV
            - In this project, it is located in '/datasets/components.csv'
            - Structured in such a way -> ['solvent_name';'solvent_id';'smiles_can']
                - solvent_name    : Full chemical name
                - solvent_id      : Unique integer identifier
                - smiles_can      : The canonical SMILES string
            - Possible to extend with new components, 
              given the known SMILES string, as every necessary molecular
              info are always parsed with RDKit internally.
        2. Raw VLE data CSV
            - In this project, it is located in '/datasets/dataset.csv'
            - Structured in such a way -> ['solv_i_id','molefrac_i','ln_gamma_i']
                - solv_i_id     : String representing the mixture components,               e.g. "solvent_587 / solvent_604", "solvent_413 / solvent_708 / solvent_716"
                - molefrac_i    : Mole fractions for respective solv_i,                     e.g. "0.10 / 0.90", "0.33 / 0.33 / 0.34"
                - ln_gamma_i       : Ground truth activity coefficient for respective solv_i,  e.g. "0.471759350 / 0.000251480", "1.719582080 / 2.044134780 / 0.887302160"
    """
    def __init__(self, components_csv: str):
        """
        Initialize pipeline and load component metadata.
        
        Args:
            components_csv (str): Path to component registry CSV file
        """
        self.components = self.load_components(components_csv)
        self.mol_cache = {}

    def load_components(self, filepath: str) -> pd.DataFrame:
        """
        Load component registry CSV containing chemical metadata.
        
        Args:
            filepath (str): Path to components CSV file
            
        Returns:
            pd.DataFrame: Component registry
            
        Raises:
            ValueError: If required columns are missing
        """
        components_df = pd.read_csv(filepath, sep=',')
        
        # Validate required columns
        required_cols = ['solvent_name','solvent_id','smiles_can']
        if not all(col in components_df.columns for col in required_cols):
            raise ValueError(f"Components CSV must contain columns: {required_cols}")
        
        # Create lookup dictionaries for fast access
        self.solvent_id_to_smiles = dict(zip(components_df['solvent_id'], components_df['smiles_can']))
        self.solvent_id_to_name = dict(zip(components_df['solvent_id'], components_df['solvent_name']))
        
        print(f"Loaded {len(components_df)} components from registry")
        return components_df

    def parse_raw_data(self, raw_csv: str) -> pd.DataFrame:
        """
        Load raw experimental CSV and detect composition convention.
        Args:
            raw_csv (str): Path to raw experimental data CSV 
        Returns:
            pd.DataFrame: Raw data
        """
        raw_data_df = pd.read_csv(raw_csv, sep=',')
        
        return raw_data_df

    def parse_systems(self, raw_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse system strings into component lists. Preserves original order.
        """
        system_parsed_df = raw_data_df.copy()
        
        def parse_system_string(system_str: str) -> List[str]:
            return [comp.strip() for comp in system_str.split('/')]
        
        system_parsed_df['component_list'] = system_parsed_df['solv_i_id'].apply(parse_system_string)
        
        return system_parsed_df

    def parse_numlist(self, system_parsed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses string-based numeric lists (molefractions, ln_gammas) into 
        lists of floats.
        
        Args:
            system_parsed_df (pd.DataFrame): DataFrame from parse_systems
            
        Returns:
            pd.DataFrame: Data with 'molefrac_list' and 'ln_gamma_list'
        """
        num_df = system_parsed_df.copy()
        
        splitter = lambda s: [float(part.strip()) for part in s.split(' / ')]

        # Apply the parsing to the columns
        num_df['molefrac_list'] = num_df['molefrac_i'].apply(splitter)
        num_df['ln_gamma_list'] = num_df['ln_gamma_i'].apply(splitter) 
        
        return num_df

    def assign_system_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate System IDs based on the SET of components (permutation invariant),
        but strictly preserve the original order of data columns.
        Logic:
        1. Row: ['Water', 'Ethanol'] -> ID based on sorted {'Ethanol', 'Water'}
        2. Row: ['Ethanol', 'Water'] -> ID based on sorted {'Ethanol', 'Water'}
        
        Result: Both get the same System ID, but input tensors remain as raw data.
        """
        df_ids = df.copy()

        # Create a temporary column just for ID generation
        # We sort the list of names ONLY to establish identity
        df_ids['sorted_component_set'] = df_ids['component_list'].apply(
            lambda x: ' / '.join(sorted(x))
        )
        
        # Map unique sorted strings to integers
        unique_systems = df_ids['sorted_component_set'].unique()
        system_id_map = {sys: i for i, sys in enumerate(unique_systems)}
        
        # Assign the ID
        df_ids['system_id'] = df_ids['sorted_component_set'].map(system_id_map)
        
        # Drop the temporary column
        df_ids.drop(columns=['sorted_component_set'], inplace=True)
        
        print(f"System ID assignment complete. Found {len(unique_systems)} unique systems.")
        return df_ids

    def construct_graphs(self, canonical_df: pd.DataFrame) -> List[Data]:
        """
        Construct PyTorch Geometric graph objects from molecular data.
        
        Converts SMILES to molecular graphs and combines multiple molecules
        into disjoint union graphs per system with attached properties.
        
        Args:
            canonical_df (pd.DataFrame): Canonicalize data with canonical compositions
            
        Returns:
            List[Data]: List of PyTorch Geometric Data objects
        """
        graph_objects = []
        
        # Diagnostic counters
        skip_reasons = {
            'no_smiles': 0,
            'invalid_mol': 0,
            'no_components': 0
        }
        skipped_systems = []
        
        # Pre-convert mol_cache values to graphs once
        graph_cache = {}
        for smiles, mol in self.mol_cache.items():
            if mol is not None:
                graph_cache[smiles] = self._mol_to_graph(mol)
        
        # Vectorize system type determination
        component_counts = canonical_df['component_list'].apply(len)
        system_types = component_counts.map({
            2: "Binary mixture",
            3: "Ternary mixture"
        }).fillna(component_counts.astype(str) + " components mixture")
        
        for (idx, row), system_type in zip(canonical_df.iterrows(), system_types):
            components = row['component_list']
            mole_fracs = row['molefrac_list']
            ln_gammas = row['ln_gamma_list']
            
            node_features_list = []
            edge_index_list = []
            edge_attr_list = []
            mol_batch_list = []
            node_offset = 0
            
            components_processed = 0
            
            for mol_idx, (comp, mole_frac) in enumerate(zip(components, mole_fracs)):
                smiles = self.solvent_id_to_smiles.get(comp)
                if not smiles:
                    skip_reasons['no_smiles'] += 1
                    print(f"Row {idx}: Missing SMILES for component '{comp}'")
                    continue
                
                # Check cache or compute
                if smiles in graph_cache:
                    node_features, edge_index, edge_attr = graph_cache[smiles]
                else:
                    if smiles not in self.mol_cache:
                        self.mol_cache[smiles] = Chem.MolFromSmiles(smiles)
                    mol = self.mol_cache[smiles]
                    if mol is None:
                        skip_reasons['invalid_mol'] += 1
                        print(f"Row {idx}: Invalid molecule for SMILES '{smiles}' (component '{comp}')")
                        continue
                    node_features, edge_index, edge_attr = self._mol_to_graph(mol)
                    graph_cache[smiles] = (node_features, edge_index, edge_attr)
                
                components_processed += 1
                
                # Clone tensors to avoid in-place modification issues
                edge_index = edge_index.clone() + node_offset
                
                node_features_list.append(node_features)
                edge_index_list.append(edge_index)
                if edge_attr is not None:
                    edge_attr_list.append(edge_attr)
                
                num_atoms = node_features.shape[0]
                mol_batch_list.append(torch.full((num_atoms,), mol_idx, dtype=torch.long))
                node_offset += num_atoms
            
            if not node_features_list:
                skip_reasons['no_components'] += 1
                skipped_systems.append({
                    'idx': idx,
                    'system_id': row['system_id'],
                    'components': components,
                    'components_attempted': len(components),
                    'components_processed': components_processed
                })
                continue
            
            x = torch.cat(node_features_list, dim=0)
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
            mol_batch = torch.cat(mol_batch_list, dim=0)
            real_names = [self.solvent_id_to_name.get(c, c) for c in components]
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mol_batch=mol_batch,
                component_batch=torch.arange(len(components), dtype=torch.long),
                component_names=components,
                actual_names=real_names,
                component_mole_frac=torch.tensor(mole_fracs, dtype=torch.float),
                component_ln_gammas=torch.tensor(ln_gammas, dtype=torch.float),
                system_type=system_type,
                system_id=row['system_id']
            )
            graph_objects.append(graph_data)
        
        # Print diagnostic summary
        print("\n=== DIAGNOSTIC SUMMARY ===")
        print(f"Total input rows: {len(canonical_df)}")
        print(f"Graphs created: {len(graph_objects)}")
        print(f"Graphs missing: {len(canonical_df) - len(graph_objects)}")
        print(f"\nSkip reasons:")
        print(f"  - Systems with no valid components: {skip_reasons['no_components']}")
        print(f"  - Component instances missing SMILES: {skip_reasons['no_smiles']}")
        print(f"  - Component instances with invalid molecules: {skip_reasons['invalid_mol']}")
        print(f"\nFirst 10 skipped systems:")
        for sys in skipped_systems[:10]:
            print(f"  Row {sys['idx']}, System {sys['system_id']}: {sys['components']}")
        
        return graph_objects
    
    def _mol_to_graph(self, mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert RDKit molecule to graph representation with rich features.

        Node features include:
        - One-hot atom identity     -> size 11  type: int('H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I')
        - Hybridization             -> size 3   type: int (sp, sp2, sp3)
        - Aromaticity               -> size 1   type: int
        - Ring membership           -> size 1   type: int
        - Hydrogen donor flags      -> size 1   type: int
        - Hydrogen acceptor flags   -> size 1   type: int
        - Formal charge             -> size 1   type: int
        - Partial charge            -> size 1   type: float
        - Atomic mass               -> size 1   type: float
        - Van der Waals radius      -> size 1   type: float
        - Degree                    -> size 1   type: int

        Edge features:
        - Bond type                 -> size 4   type: float (single, double, triple, aromatic)

        Returns:
            node_features: torch.Tensor [num_atoms, num_node_features]
            edge_index: torch.Tensor [2, num_edges]
            edge_attr: torch.Tensor [num_edges, num_edge_features]
        """
        # ATOMIC DEFINITIONS
        pt = Chem.GetPeriodicTable()
        mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol)
        atomic_types = ['H', 'C', 'N', 'O', 
                        'F', 'Si', 'P', 'S', 
                        'Cl', 'Br', 'I']
        node_features = []

        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()

            # ATOMIC IDENTITY
            # 1. Atom type one-hot
            atom_type_vec = [1 if atom_type == el else 0 for el in atomic_types]

            # 2. Hybridization one-hot
            hyb = atom.GetHybridization()
            if hyb == Chem.rdchem.HybridizationType.SP3:
                hyb_vec = [1, 0, 0]
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                hyb_vec = [0, 1, 0]
            elif hyb == Chem.rdchem.HybridizationType.SP:
                hyb_vec = [0, 0, 1]
            else:
                hyb_vec = [0, 0, 0]

            # 3. Aromaticity flag
            aromaticity = [1 if atom.GetIsAromatic() else 0]

            # 4. Ring membership flag
            in_ring = [1 if atom.IsInRing() else 0]

            # 5. Hydrogen bonding donor flag
            hydrogen_donor = [1 if atom_type in ['N', 'O','F'] and any(nbr.GetSymbol() == 'H' for nbr in atom.GetNeighbors()) else 0]

            # 6. Hydrogen bonding acceptor flag
            hydrogen_acceptor = [1 if atom_type in ['N', 'O', 'F'] else 0]

            # 7. Formal charge
            formal_charge = [atom.GetFormalCharge()]

            # 8. Partial charge
            g_charge = atom.GetDoubleProp('_GasteigerCharge')
            if np.isnan(g_charge) or np.isinf(g_charge):
                g_charge = 0.0
            partial_charge = [g_charge]

            # 9. Atomic mass
            mass = [atom.GetMass() * 0.01]

            # 10. Van der waals radius
            vdw_radius = [pt.GetRvdw(atom.GetAtomicNum())]

            # 11. Degree
            degree = [atom.GetTotalDegree()]
            # Combine features
            features = (atom_type_vec + 
                        hyb_vec + aromaticity + in_ring +
                         hydrogen_donor + hydrogen_acceptor 
                         + formal_charge + partial_charge 
                         + mass + vdw_radius + degree
                         )
            node_features.append(features)

        node_features = torch.tensor(node_features, dtype=torch.float)

        # --- Edge features ---
        edge_index = []
        edge_attr = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_vec = [1 if bond.GetBondType() == b_type else 0 for b_type in bond_types]
            edge_attr.append(bond_vec)
            edge_attr.append(bond_vec)

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(bond_types)), dtype=torch.float)

        return node_features, edge_index, edge_attr

    def save_canonical_df(self, canonical_df: pd.DataFrame, filepath: str):
        """
        Save canonical DataFrame to CSV with semicolon separator.

        Args:
            canonical_df (pd.DataFrame): Processed canonical DataFrame
            filepath (str): Path to save CSV file
        """
        canonical_df.to_csv(filepath, sep=',', index=False)
        print(f"Canonical data saved to: {filepath}")

    def get_system_summary(self, canonical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each unique system.
        
        Args:
            canonical_df (pd.DataFrame): Canonical data
            
        Returns:
            pd.DataFrame: Summary statistics per system
        """
        summary = canonical_df.groupby('canonical_system').agg({
            'system_id': 'first'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        return summary.reset_index()
    
    def run_pipeline(
        self,
        raw_csv: str,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, List[Data]]:
        
        if verbose: print("="*60 + "\nEXECUTION STARTED\n" + "="*60)

        # 1. Parse Raw Data
        if verbose: print("\nStep 1: Parsing raw data...")
        raw_df = self.parse_raw_data(raw_csv)

        # 2. Parse Strings to Lists (Keep Order!)
        if verbose: print("\nStep 2: Parsing system strings (preserving order)...")
        parsed_df = self.parse_systems(raw_df)

        # 3. Parse Numbers to Floats (Keep Order!)
        if verbose: print("\nStep 3: Parsing numerical values...")
        num_df = self.parse_numlist(parsed_df)

        # 4. Assign IDs based on Set Identity (A+B = B+A)
        if verbose: print("\nStep 4: Assigning permutation-invariant System IDs...")
        # This replaces canonicalize_rows
        ready_df = self.assign_system_ids(num_df)

        # 5. Construct Graphs (Uses original order)
        if verbose: print("\nStep 5: Constructing molecular graphs...")
        graphs = self.construct_graphs(ready_df)

        if verbose:
            print(f"\nPIPELINE COMPLETE! Created {len(graphs)} graphs.")
            print(f"Unique systems: {ready_df['system_id'].nunique()}")

        return ready_df, graphs