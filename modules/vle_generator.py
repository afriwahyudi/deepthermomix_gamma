import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from torch_geometric.data import Data
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup
import re
from contextlib import closing

# --- 1. Thermodynamics Helper: Antoine Equation ---
class AntoineEquation:
    """
    Calculates Saturation Pressure (P_sat) in Bar.
    Equation: log10(P_sat) = A - (B / (T + C))
    
    Robustly scrapes NIST WebBook by:
      1. Attempting InChI search (precise).
      2. Attempting Name search (fallback).
      3. Extracting the official Name from the NIST page for plotting.
    """
    def __init__(self):
        self.cache = {} # Stores params
        self.names = {} # Stores scraped names
        self.base_url = "https://webbook.nist.gov/cgi/cbook.cgi"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })

    def _get_inchi(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToInchi(mol)
        return None
    
    def _canonicalize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles

    def _fetch_nist_params(self, smiles, name=None):
        """Scrapes NIST WebBook for Antoine parameters and Name."""
        
        inchi = self._get_inchi(smiles)
        entries = []
        scraped_name = None
        
        # 1. Try InChI Search
        if inchi:
            print(f"Attempting NIST InChI Search for: {smiles}")
            entries, scraped_name = self._query_nist({'InChI': inchi, 'Units': 'SI', 'Mask': '4'})
        
        # 2. Try Name Search (if InChI failed)
        if not entries and name:
            print(f"InChI search empty. Retrying with Name: {name}")
            entries, scraped_name = self._query_nist({'Name': name, 'Units': 'SI', 'Mask': '4'})

        if entries:
            print(f"  -> Success: Found {len(entries)} parameter sets.")
            if scraped_name:
                print(f"  -> Identified as: {scraped_name}")
                self.names[smiles] = scraped_name
        else:
            print(f"  -> Failed: No parameters found for {smiles} (Name: {name}).")
            
        return entries

    def _query_nist(self, params):
        entries = []
        scraped_name = None
        try:
            # Create the query URL for logging/debugging
            req = requests.Request('GET', self.base_url, params=params)
            prepped = self.session.prepare_request(req)
            print(f"  -> Querying: {prepped.url}")
            
            response = self.session.send(prepped, timeout=15)
            
            if response.status_code != 200:
                print(f"  -> HTTP Error {response.status_code}")
                return [], None
                
            content = response.content
        except Exception as e:
            print(f"  -> Network error: {e}")
            return [], None

        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract Name from Header (Usually <h1 id="Top">Name</h1>)
        header = soup.find('h1', id='Top')
        if header:
            scraped_name = header.get_text().strip()

        # Find the Antoine table. 
        tables = soup.find_all('table')
        target_tables = []
        
        for t in tables:
            if t.get('aria-label') == 'Antoine Equation Parameters':
                target_tables.append(t)
            elif t.find_previous_sibling('h3') and 'Antoine' in t.find_previous_sibling('h3').text:
                 target_tables.append(t)

        for table in target_tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                # We need at least 4 columns: T-range, A, B, C
                if len(cols) >= 4:
                    try:
                        # 1. Parse Temperature Range
                        temp_text = cols[0].get_text().strip()
                        temp_matches = re.findall(r"[-+]?\d*\.\d+|\d+", temp_text)
                        
                        if len(temp_matches) >= 2:
                            t_min, t_max = float(temp_matches[0]), float(temp_matches[1])
                        elif len(temp_matches) == 1:
                            val = float(temp_matches[0])
                            t_min, t_max = val - 10, val + 10 
                        else:
                            t_min, t_max = 0.0, 5000.0 

                        # 2. Parse Coefficients
                        val_A = float(cols[1].get_text().replace(' ',''))
                        val_B = float(cols[2].get_text().replace(' ',''))
                        val_C = float(cols[3].get_text().replace(' ',''))
                        
                        entries.append({
                            'A': val_A, 'B': val_B, 'C': val_C,
                            't_min': t_min, 't_max': t_max
                        })
                    except (ValueError, IndexError):
                        continue
        return entries, scraped_name

    def get_Psat(self, smiles, T_kelvin, name=None):
        # 1. Check Cache
        if smiles in self.cache:
            params_list = self.cache[smiles]
        else:
            # 2. Scrape
            params_list = self._fetch_nist_params(smiles, name)
            self.cache[smiles] = params_list
        
        if not params_list:
            print(f"Warning: No Antoine params found for {smiles}. VLE will be inaccurate (Using P=1.0 bar).")
            return 1.0

        # 3. Select best parameters
        best_params = None
        
        # Priority 1: T is strictly within range
        for p in params_list:
            if p['t_min'] <= T_kelvin <= p['t_max']:
                best_params = p
                break
        
        # Priority 2: Closest range (Extrapolation)
        if best_params is None:
            def dist_to_range(p):
                if T_kelvin < p['t_min']: return p['t_min'] - T_kelvin
                if T_kelvin > p['t_max']: return T_kelvin - p['t_max']
                return 0
            best_params = min(params_list, key=dist_to_range)
            print(f"  -> Note: T={T_kelvin}K is outside NIST range ({best_params['t_min']}-{best_params['t_max']}). Extrapolating.")

        A, B, C = best_params['A'], best_params['B'], best_params['C']
        log_p = A - (B / (T_kelvin + C))
        return 10**log_p
    
    def get_stored_name(self, smiles):
        """Retrieve name scraped from NIST if available."""
        return self.names.get(smiles, None)


# --- 2. The VLE Generation Class ---
class VLEAnalyzer:
    def __init__(self, model, pipeline, device='cpu'):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.antoine = AntoineEquation()
        self.model.to(device)
        self.model.eval()
        
        # Map Canonical SMILES -> Name for fallback search
        self.smiles_map = {}
        if hasattr(pipeline, 'solvent_id_to_smiles'):
            for sid, smi in pipeline.solvent_id_to_smiles.items():
                name = pipeline.solvent_id_to_name.get(sid, None)
                if name and smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                        self.smiles_map[can_smi] = name

    def _prepare_single_point(self, smiles_list, mole_fracs):
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        mol_batch_list = []
        node_offset = 0
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
                
            x, edge_index, edge_attr = self.pipeline._mol_to_graph(mol)
            edge_index = edge_index + node_offset
            
            node_features_list.append(x)
            edge_index_list.append(edge_index)
            if edge_attr is not None:
                edge_attr_list.append(edge_attr)
            
            num_atoms = x.shape[0]
            mol_batch_list.append(torch.full((num_atoms,), i, dtype=torch.long))
            node_offset += num_atoms

        x = torch.cat(node_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
        mol_batch = torch.cat(mol_batch_list, dim=0)
        
        # FIX: Made component_batch_batch dynamic based on number of components
        num_comps = len(smiles_list)
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, mol_batch=mol_batch,
            component_batch_batch=torch.zeros(num_comps, dtype=torch.long),
            component_mole_frac=torch.tensor(mole_fracs, dtype=torch.float)
        )
        return data.to(self.device)

    def generate_isotherm(self, smiles1, smiles2, T_kelvin, steps=21):
        x1_range = np.linspace(0, 1, steps)
        results = []
        R = 8.31446261815324 # J/mol/K

        # 1. Resolve Initial Names from Pipeline (User Provided)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        can1 = Chem.MolToSmiles(mol1, isomericSmiles=True) if mol1 else smiles1
        can2 = Chem.MolToSmiles(mol2, isomericSmiles=True) if mol2 else smiles2

        pipeline_name1 = self.smiles_map.get(can1, None)
        pipeline_name2 = self.smiles_map.get(can2, None)

        # 2. Fetch Saturation Pressures (Triggers Scraping)
        Psat1 = self.antoine.get_Psat(smiles1, T_kelvin, name=pipeline_name1)
        Psat2 = self.antoine.get_Psat(smiles2, T_kelvin, name=pipeline_name2)
        
        # 3. Determine Final Display Names (Prefer Pipeline, then NIST Scraped, then SMILES)
        name1 = pipeline_name1 or self.antoine.get_stored_name(smiles1) or smiles1
        name2 = pipeline_name2 or self.antoine.get_stored_name(smiles2) or smiles2

        print(f"Generating VLE for {name1} / {name2} at {T_kelvin}K")
        print(f"Psat1: {Psat1:.4f} bar, Psat2: {Psat2:.4f} bar")

        for x1 in x1_range:
            x2 = 1.0 - x1
            data = self._prepare_single_point([smiles1, smiles2], [x1, x2])

            ln_gamma_pred, _, _ = self.model(data)
            
            ln_g1 = ln_gamma_pred[0].item()
            ln_g2 = ln_gamma_pred[1].item()
            g1 = np.exp(ln_g1)
            g2 = np.exp(ln_g2)

            p1_partial = x1 * g1 * Psat1
            p2_partial = x2 * g2 * Psat2
            P_total = p1_partial + p2_partial
            
            y1 = p1_partial / P_total if P_total > 1e-9 else 0.0
            if x1 > 0.999: y1 = 1.0
            if x1 < 0.001: y1 = 0.0

            g_excess =  (R * T_kelvin) * (x1 * ln_g1 + x2 * ln_g2)

            g_reduced = g_excess /( R * T_kelvin)

            results.append({
                'x1': x1, 'y1': y1, 'P': P_total,
                'ln_gamma1': ln_g1, 'ln_gamma2': ln_g2,
                'gamma1': g1, 'gamma2': g2, 'g_excess': g_excess,
                'g_reduced': g_reduced
            })
        
        df = pd.DataFrame(results)
        # Embed names into DataFrame attributes for the plotter
        df.attrs['name1'] = name1
        df.attrs['name2'] = name2
        return df

    def plot_vle(self, df, title_prefix=None):
        # Retrieve names from dataframe metadata if available
        name1 = df.attrs.get('name1', 'Component 1')
        name2 = df.attrs.get('name2', 'Component 2')
        
        if title_prefix is None:
            title_prefix = f"{name1} / {name2}"

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        ax[0].plot(df['x1'], df['P'], 'b', label=f'$x$ ($x_{{{name1}}}$)', linewidth=2.5)
        ax[0].plot(df['y1'], df['P'], 'r', label=f'$y$ ($y_{{{name1}}}$)', linewidth=2.5)
        ax[0].set_xlabel(f'$z$ ${name1}$')
        ax[0].set_ylabel('Pressure (bar)')
        ax[0].set_title(f'{title_prefix}: P-x-y Diagram')
        ax[0].set_xlim(0,1)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(df['x1'], df['ln_gamma1'], 'b', label=f'ln($\gamma_{{{name1}}}$)', linewidth=2.5)
        ax[1].plot(df['x1'], df['ln_gamma2'], 'r', label=f'ln($\gamma_{{{name2}}}$)', linewidth=2.5)
        ax[1].set_xlabel(f'$z$ ${name1}$')
        ax[1].set_ylabel('ln($\gamma$)')
        ax[1].set_title(f'{title_prefix}: Activity Coefficients')
        ax[1].set_xlim(0,1)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        ax[1].axhline(0, color='red', linewidth=1.0, linestyle='--')

        ax[2].plot(df['x1'], df['g_reduced'], 'k-o', label='$G^E / RT$')
        ax[2].set_xlabel(f'$z$ ${name1}$')
        ax[2].set_ylabel('$G^E / RT$')
        ax[2].set_title(f'{title_prefix}: Excess Gibbs Energy')
        ax[2].set_xlim(0,1)
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        ax[2].axhline(0, color='red', linewidth=1.0, linestyle='--')
        
        plt.tight_layout()
        plt.show()