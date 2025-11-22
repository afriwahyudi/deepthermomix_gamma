import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from modules.loss_func import GibbsDuhemLoss, MixMSELoss

class ComputeMetric:
    def __init__(self, model, loader, device='cuda'):
        self.model = model
        self.loader = loader
        self.device = device
        self.datadriven_loss_fn = MixMSELoss()
        self.gd_loss_fn = GibbsDuhemLoss()

    def run_evaluation(self):
        """
        Returns:
            df_formatted (pd.DataFrame): String format (val1 / val2) for CSVs.
            df_raw       (pd.DataFrame): Numeric format for visualization.
            metrics      (tuple): rmse, mae, r2
        """
        self.model.eval()
        self.model.to(self.device)
        
        # Fast List Storage
        list_y_true = []
        list_y_pred = []
        list_x_mole = []
        list_sys_id = []
        list_group_id = [] 
        running_gd_loss = []
        
        batch_offset = 0 

        print(f"Starting evaluation on {len(self.loader)} batches...")

        for batch in self.loader:
            batch = batch.to(self.device)
            
            # 1. Forward
            y_pred_batch, _, _ = self.model(batch)
            
            # 2. GD Loss
            gd_loss = self.gd_loss_fn(batch, y_pred_batch)
            running_gd_loss.append(gd_loss.item())
            
            # 3. Extract Data (Flattened)
            y_true = batch.component_ln_gammas.detach().cpu().numpy().flatten()
            y_pred = y_pred_batch.detach().cpu().numpy().flatten()
            x_mole = batch.component_mole_frac.detach().cpu().numpy().flatten()
            
            # 4. Grouping IDs (Global Batch Index)
            local_batch_idx = batch.component_batch_batch.detach().cpu().numpy().flatten()
            global_batch_idx = local_batch_idx + batch_offset
            
            # 5. System Names/IDs
            if hasattr(batch, 'system_id'):
                # Use system_id if available
                sys_ids = batch.system_id.detach().cpu().numpy()[local_batch_idx]
                list_sys_id.append(sys_ids)
            elif hasattr(batch, 'actual_names'):
                 # Use string names if available
                 names = np.array(batch.actual_names)
                 sys_ids = names[local_batch_idx]
                 list_sys_id.append(sys_ids)
            
            list_y_true.append(y_true)
            list_y_pred.append(y_pred)
            list_x_mole.append(x_mole)
            list_group_id.append(global_batch_idx)
            
            batch_offset += batch.num_graphs

        # --- Concatenate ---
        y_true_all = np.concatenate(list_y_true)
        y_pred_all = np.concatenate(list_y_pred)
        x_mole_all = np.concatenate(list_x_mole)
        group_id_all = np.concatenate(list_group_id)
        
        if list_sys_id:
            sys_id_all = np.concatenate(list_sys_id)
        else:
            sys_id_all = group_id_all # Fallback

        # --- Metrics (Pure Numpy) ---
        rmse_log = np.sqrt(np.mean((y_true_all - y_pred_all)**2))
        mae_log = np.mean(np.abs(y_true_all - y_pred_all))
        r2_log = r2_score(y_true_all, y_pred_all)
        avg_gd_loss = np.mean(running_gd_loss)

        print("\n=== Global Evaluation Results ===")
        print(f"RMSE (ln gamma): {rmse_log:.5f}")
        print(f"MAE  (ln gamma): {mae_log:.5f}")
        print(f"R    (ln gamma): {r2_log:.5f}")
        print(f"GD Loss:    {avg_gd_loss:.6e}")

        # --- Create Raw Numeric DataFrame (For Visualization) ---
        df_raw = pd.DataFrame({
            'group_id': group_id_all,
            'solv_i_id': sys_id_all,
            'molefrac_i': x_mole_all,
            'ln_gamma_exp': y_true_all,
            'ln_gamma_pred': y_pred_all
        })
        # Add Linear Error for convenient plotting later
        df_raw['error_lin'] = np.exp(df_raw['ln_gamma_pred']) - np.exp(df_raw['ln_gamma_exp'])

        # --- Create Formatted DataFrame (For CSV) ---
        df_formatted = self._format_to_string(df_raw)
        
        return df_formatted, df_raw, rmse_log, mae_log, r2_log

    def _format_to_string(self, df_raw):
        """
        Internal helper: Collapses the raw numeric rows into slash-separated strings.
        """
        # Define string formatter helper
        def join_fmt(x, fmt):
            return " / ".join([fmt.format(val) for val in x])

        # GroupBy using the unique group_id
        df_str = df_raw.groupby('group_id').agg({
            'solv_i_id': 'first',
            'molefrac_i': lambda x: join_fmt(x, "{:.2f}"),
            'ln_gamma_exp': lambda x: join_fmt(x, "{:.9f}"),
            'ln_gamma_pred': lambda x: join_fmt(x, "{:.9f}")
        }).reset_index(drop=True)

        return df_str