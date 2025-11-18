import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch_scatter import scatter_add, scatter_mean

class MixMSELoss(nn.Module):
    """
    Mean of Mixture Sum-of-Squares Loss

    Computes the loss by first calculating the Sum of Squared Errors (SSE)
    for each mixture (datapoint), and then taking the mean of those SSEs.

    This loss is suitable for benchmarking, as it does not normalize
    by the number of components in each mixture.

    Equations:
        Let m = mixture index (1 to M)
        Let i = component index (1 to N for mixture m)
        
        TSE_m = SUM_i [ (y_pred_m,i - y_true_m,i)^2 ]
        Loss = (1/M) * SUM_m [ TSE_m ]
    """
    def __init__(self):
        super(MixMSELoss, self).__init__()

    def forward(self, y_pred, batched_data):
        """
        Calculates the forward pass of the loss.

        Args:
            y_pred (torch.Tensor): Predictions from the model, 
                                   shape [N_total_components]
            batched_data (torch_geometric.data.Batch): 
                                   The batch object from the DataLoader.
                                   Must contain 'component_gammas' (as y_true)
                                   and 'component_batch_batch' (as batch_index).
        """
        
        # Extract targets and batch index from the batch object
        y_true = batched_data.component_gammas
        batch_index = batched_data.component_batch_batch

        # 1. Calculate the component-wise squared error for all N components
        squared_errors = torch.pow(y_pred - y_true, 2)
        
        # 2. Calculate the inner sum: SUM_i [ ... ]
        summed_mixture_errors = scatter_add(
            squared_errors, 
            batch_index, 
            dim=0
        )
        
        # 3. Calculate the outer mean: (1/M) * SUM_m [ ... ]
        loss = torch.mean(summed_mixture_errors)
        
        return loss
    
# Gibbs Duhem Loss Module
class GibbsDuhemLoss(nn.Module):
    """
    Computes the Gibbs-Duhem constraint loss.

    From Gmehling et al., Chemical Thermodynamics
    for Process Simulation, First Edition, pg. 162:
    
        gbar_i^E    = RT * ln(gamma_i)                                               . . . . (4.85)
        g^E         = RT * sum_i(x_i * ln(gamma_i))                                  . . . . (4.86)

        Fundamental equation:

        s^E dT - v^E dP + sum(x_i * d(gbar_i^E))       = 0                           . . . . (4.36) 
        
        At constant T and P:
            sum(x_i * d(gbar_i^E) = 0                                                . . . . (1)
        
        Translating the total differential into partial derivatives:
            d(gbar_i^E) = sum_j(∂(gbar_i^E) / ∂(x_j)) * dx_j                         . . . . (2)

        Expanded with respect to all N dependent mole fractions (x_j):
            v_j = sum_i(x_i * ∂(gbar_i^E) / ∂(x_j)) = c  (for all j = 1...N)         . . . . (3)
               
    Description:
        This loss function enforces the Gibbs-Duhem differential constraint
        ln_gamma_calc from the forward pass
        This allows retrival of gbar_i^E from Eq. (4.85)
        and be used for interpretation.

    Args:
        loss_type (str): 'explicit' or 'optimized'. Defaults: 'optimized'.
            - 'explicit' : Slow, explicit implementation.
            - 'optimized': Fast, batched implementation.
    """

    def __init__(self, loss_type='optimized', track_graph=True):
        super(GibbsDuhemLoss, self).__init__()
        self.loss_type = loss_type
        self.track_graph = track_graph

    def forward(self, data, ln_gamma_calc):
        """
        Args:
            data: PyG Data object with component_mole_frac and component_batch_batch
            ln_gamma_calc: [num_components_total] from DeepThermoMix
        
        Returns:
            gd_loss: Scalar loss enforcing Gibbs-Duhem constraint
        """

        mole_frac = data.component_mole_frac
        component_batch = data.component_batch_batch
        T = 298.15
        R = 8.31446261815324   
        ln_gamma = ln_gamma_calc.sum(dim=-1)     
        g_excess = ln_gamma * (R * T)

        if self.loss_type == 'explicit':
            gd_loss_batch = []

            for batch_idx in torch.unique(component_batch):
                mask = (component_batch == batch_idx)
                indices = torch.where(mask)[0]
                
                g_excess_i = g_excess[indices]
                num_components = len(indices)
                
                jacobian_rows = []
                for i in range(num_components):
                    full_grad = autograd.grad(
                        outputs=g_excess_i[i],
                        inputs=mole_frac,
                        retain_graph=True,
                        create_graph=self.track_graph
                    )[0]
                    
                    batch_grad = full_grad[indices]
                    jacobian_rows.append(batch_grad)
                
                jacobian = torch.stack(jacobian_rows)
                
                x_i = mole_frac[indices]
                
                vj = torch.matmul(x_i.unsqueeze(0), jacobian).squeeze()
                vj_mean = torch.mean(vj)
                gd_loss_sample = torch.sum((vj - vj_mean) ** 2)
                gd_loss_batch.append(gd_loss_sample)

            gd_loss = torch.mean(torch.stack(gd_loss_batch))
        
        elif self.loss_type == 'optimized':
            
            S = mole_frac * g_excess
            S_per_sample = scatter_add(S, component_batch, dim=0)
            
            (full_grad,) = autograd.grad(
                outputs=torch.sum(S_per_sample),
                inputs=mole_frac,
                retain_graph=True,
                create_graph=self.track_graph
            )
            
            vj = full_grad - g_excess
            
            vj_mean_per_sample = scatter_mean(vj, component_batch, dim=0)
            vj_mean_expanded = vj_mean_per_sample[component_batch]
            
            gd_loss_sample = (vj - vj_mean_expanded) ** 2

            gd_loss_batch = scatter_add(gd_loss_sample, component_batch, dim=0)
            
            gd_loss = torch.mean(gd_loss_batch)

        return gd_loss