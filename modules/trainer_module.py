import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import optuna
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_add, scatter_mean

from modules.loss_func import GibbsDuhemLoss, MixMSELoss

class DTMPNNTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        include_gd,
        device='cuda',
        lr=1e-6,
        weight_decay=1e-5,
        data_driven_weight=1.0,
        gd_weight=0.1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.data_driven_weight = data_driven_weight
        self.gd_weight = gd_weight
        
        self.include_gd = include_gd
        self.datadriven_loss_fn = MixMSELoss()
        self.gd_loss_fn = GibbsDuhemLoss(track_graph=self.include_gd) 
        
        self.datadriven_loss_name = self.datadriven_loss_fn.__class__.__name__.replace("Loss", "").upper()
        
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.history = {
            'train_loss': [], 'train_data_driven': [], 'train_gd': [],
            'train_rmse': [], 'train_mae'        : [], 
            'val_loss'  : [], 'val_data_driven'  : [], 'val_gd': [],
            'val_rmse'  : [], 'val_mae'          : [], 
            'lr': []
        }
        
        self.log_buffer = [] 
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _log(self, message, print_to_console=False):
        self.log_buffer.append(message)
        if print_to_console:
            print(message)
            
    def compute_losses(self, batched_data):
        y_pred_batch, _, _ = self.model(batched_data)
        
        data_driven_loss = self.datadriven_loss_fn(y_pred_batch, batched_data)
        
        loss_gd = self.gd_loss_fn(batched_data, y_pred_batch)
        
        if self.include_gd:
            total_loss = self.data_driven_weight * data_driven_loss + self.gd_weight * loss_gd
        else:
            total_loss = self.data_driven_weight * data_driven_loss
                
        y_true = batched_data.component_ln_gammas.detach().cpu().numpy()
        y_pred = y_pred_batch.detach().cpu().numpy()
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        return total_loss, data_driven_loss, loss_gd, rmse, mae
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = epoch_data_driven_loss = epoch_gd = 0.0
        epoch_rmse = epoch_mae  = 0.0
        
        for batched_data in tqdm(self.train_loader, desc='Training', leave=False):
            batched_data = batched_data.to(self.device)
            self.optimizer.zero_grad()
            total_loss, data_driven_loss, loss_gd, rmse, mae = self.compute_losses(batched_data)
            total_loss.backward()
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_data_driven_loss += data_driven_loss.item()
            epoch_gd += loss_gd.item()
            epoch_rmse += rmse
            epoch_mae += mae
  
        n_batches = len(self.train_loader)
        return (epoch_loss / n_batches, epoch_data_driven_loss / n_batches, epoch_gd / n_batches,
                epoch_rmse / n_batches, epoch_mae / n_batches)
        
    def validate(self, loader):
        self.model.eval()
        total_data_driven_loss = 0.0
        total_gd_loss = 0.0
        all_true = []
        all_pred = []
        
        for batched_data in loader:
            batched_data = batched_data.to(self.device)
            
            y_pred_batch, _, _ = self.model(batched_data)
            
            gd_loss = self.gd_loss_fn(batched_data, y_pred_batch)

            with torch.no_grad():
                data_driven_loss = self.datadriven_loss_fn(y_pred_batch, batched_data)
                
                total_data_driven_loss += data_driven_loss.item()
                total_gd_loss += gd_loss.item() 

                all_true.append(batched_data.component_ln_gammas.detach().cpu().numpy())
                all_pred.append(y_pred_batch.detach().cpu().numpy())
            
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        n_batches = len(loader)
        avg_data_driven_loss = total_data_driven_loss / n_batches
        avg_gd_loss = total_gd_loss / n_batches
        
        return avg_data_driven_loss, avg_data_driven_loss, avg_gd_loss, rmse, mae
        
    def train(self, epochs, save_dir='checkpoints', log_file_path='training_run_log.txt', 
              save_best=True, save_every=None, patience=None, optuna_trial=None):
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
        else:
            save_path = None
        
        self.log_buffer = [] 

        if patience:
            patience_counter = 0
            self._log(f"Starting training for {epochs} epochs with Early Stopping (patience={patience})...")
        else:
            self._log(f"Starting training for {epochs} epochs...")
        
        if optuna_trial:
            self._log(f"--- Optuna Trial {optuna_trial.number} ---")

        self._log(f"Device: {self.device}")
        self._log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self._log("-" * 70)

        FIXED_GD_WIDTH = 20
        
        for epoch in range(1, epochs + 1):
            train_loss, train_data_driven, train_gd, train_rmse, train_mae= self.train_epoch()
            val_loss, val_data_driven, val_gd, val_rmse, val_mae= self.validate(self.val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_data_driven'].append(train_data_driven)
            self.history['train_gd'].append(train_gd)
            self.history['train_rmse'].append(train_rmse)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_data_driven'].append(val_data_driven)
            self.history['val_gd'].append(val_gd)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(current_lr)
            
            train_gd_full = f"GD: {train_gd:.4e},"
            train_gd_padded = f"{train_gd_full:<{FIXED_GD_WIDTH}}"
            val_gd_full = f"GD: {val_gd:.4e}," 
            val_gd_padded = f"{val_gd_full:<{FIXED_GD_WIDTH}}"
            
            self._log(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            self._log(f"  Train - Data Loss ({self.datadriven_loss_name}): {train_data_driven:.4f}, {train_gd_padded}RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
            self._log(f"  Val   - Data Loss ({self.datadriven_loss_name}): {val_data_driven:.4f}, {val_gd_padded}RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                if patience:
                    patience_counter = 0 
                
                if save_best and save_path:
                    self.save_checkpoint(save_path / 'best_model.pt', epoch)
                    self._log(f"  New best model saved! (Val Loss: {val_loss:.4f})")
            elif patience:
                patience_counter += 1
            
            if save_every and save_path and epoch % save_every == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch}.pt', epoch)
            
            self._log("-" * 70)

            if optuna_trial:
                optuna_trial.report(val_loss, epoch) 
                
                if optuna_trial.should_prune(): 
                    try:
                        log_content = "\n".join(self.log_buffer)
                        log_path = Path(log_file_path)
                        log_path.write_text(log_content + f"\n\n--- TRIAL PRUNED AT EPOCH {epoch} ---")
                        self._log(f"--- Trial pruned at epoch {epoch}. ---", print_to_console=True)
                    except Exception as e:
                        self._log(f"--- Trial pruned at epoch {epoch}. (Log write failed: {e}) ---", print_to_console=True)
                    
                    torch.cuda.empty_cache()
                    raise optuna.TrialPruned() 

            try:
                log_content = "\n".join(self.log_buffer)
                log_path = Path(log_file_path)
                log_path.write_text(log_content) 
            except Exception as e:
                self._log(f"\n--- WARNING: Could not update log file during epoch {epoch} due to error: {e} ---", print_to_console=True)
            
            if patience and patience_counter >= patience:
                self._log(f"--- Epoch {epoch}: Early stopping triggered after {patience} epochs of no improvement. ---", print_to_console=True)
                break 
        
        self._log(f"\nTraining complete!")
        self._log(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
        if self.test_loader is not None:
            self._log("\nEvaluating on test set...")
            test_loss, test_data_driven, test_gd, test_rmse, test_mae = self.validate(self.test_loader)
            
            test_gd_full = f"GD: {test_gd:.4e},"
            test_gd_padded = f"{test_gd_full:<{FIXED_GD_WIDTH}}"
                
            self._log(f"Test Loss: {test_loss:.4f}")
            self._log(f"Test {self.datadriven_loss_name}: {test_data_driven:.4f}, {test_gd_padded}RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        else:
            self._log("\nTraining complete. No test loader provided, skipping final test evaluation.")


        try:
            log_content = "\n".join(self.log_buffer)
            log_path = Path(log_file_path)
            log_path.write_text(log_content)
            self._log(f"\n--- Full log successfully finalized and written to: {log_path.name} ---", print_to_console=True)
        except Exception as e:
            self._log(f"\n--- WARNING: Could not finalize log file due to error: {e} ---", print_to_console=True)

        return self.history
    
    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, path)
        torch.cuda.empty_cache()
    
    def load_checkpoint(self, path):
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        return checkpoint['epoch']
    
    def plot_history(self, save_path=None):
        FONTSIZE = 16 
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1) 

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(self.history['train_loss'], label='Train')
        ax0.plot(self.history['val_loss'], label='Val')
        ax0.set_xlabel('Epoch', fontsize=FONTSIZE)
        ax0.set_ylabel('Total Loss', fontsize=FONTSIZE)
        ax0.set_title('Total Loss', fontsize=FONTSIZE + 2)
        ax0.tick_params(axis='both', which='major', labelsize=FONTSIZE - 2)
        ax0.legend(fontsize=FONTSIZE - 2)
        ax0.grid(True, alpha=0.3)

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(self.history['train_data_driven'], label='Train')
        ax1.plot(self.history['val_data_driven'], label='Val')

        ax1.set_xlabel('Epoch', fontsize=FONTSIZE)
        ax1.set_ylabel(f'{self.datadriven_loss_name} Loss', fontsize=FONTSIZE)
        ax1.set_title(f'{self.datadriven_loss_name} Loss', fontsize=FONTSIZE + 1)
        ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE - 2)
        ax1.legend(fontsize=FONTSIZE - 2)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(self.history['train_gd'], label='Train GD')
        ax2.plot(self.history['val_gd'], label='Val GD', linestyle='--')
        
        ax2.set_xlabel('Epoch', fontsize=FONTSIZE)
        ax2.set_ylabel('Loss', fontsize=FONTSIZE)
        ax2.set_yscale('log')
        ax2.set_title('Gibbs-Duhem Losses', fontsize=FONTSIZE + 1)
        ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE - 2)
        ax2.legend(fontsize=FONTSIZE - 2)
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()