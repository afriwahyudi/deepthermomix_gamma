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
                                   Must contain 'component_ln_gammas' (as y_true)
                                   and 'component_batch_batch' (as batch_index).
        """
        
        # Extract targets and batch index from the batch object
        y_true = batched_data.component_ln_gammas
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
            - 'explicit' : Slow, explicit implementation    ; complexity O(N^2) per mixture
            - 'optimized': Fast, batched implementation     ; complexity O(N) per mixture
    """

    def __init__(self, loss_type='optimized', track_graph=True):
        super(GibbsDuhemLoss, self).__init__()
        self.loss_type = loss_type
        self.track_graph = track_graph

    def forward(self, data, prediction):
        """
        Args:
            data            : PyG Data object with component_mole_frac and component_batch_batch
            ln_gamma_calc   : [num_components_total] from DeepThermoMix
        
        Returns:
            gd_loss         : Scalar loss enforcing Gibbs-Duhem constraint
        """

        mole_frac = data.component_mole_frac
        component_batch = data.component_batch_batch
        T = 298.15
        R = 8.31446261815324  
        g_excess_partial = prediction * (R * T)

        if self.loss_type == 'explicit':
            gd_loss_batch = []

            for batch_idx in torch.unique(component_batch):
                mask = (component_batch == batch_idx)
                indices = torch.where(mask)[0]
                
                # Slice data for this specific mixture
                g_partial_local = g_excess_partial[indices]
                num_components = len(indices)
                
                # Construct Jacobian Matrix: J_ij = ∂(g_i^E) / ∂(x_j)
                jacobian_rows = []
                for i in range(num_components):
                    full_grad = autograd.grad(
                        outputs=g_partial_local[i],
                        inputs=mole_frac,
                        retain_graph=True,
                        create_graph=self.track_graph
                    )[0]
                    
                    local_gradients = full_grad[indices]
                    jacobian_rows.append(local_gradients)
                
                jacobian = torch.stack(jacobian_rows)
                
                # Calculate Consistency Residual (v_j)
                x_i = mole_frac[indices]
                consistency_residual = torch.matmul(x_i.unsqueeze(0), jacobian).squeeze()
                
                # Enforce Gibbs-Duhem: Variance of v_j must be 0 (Source 72)
                residual_mean = torch.mean(consistency_residual)
                gd_loss_sample = torch.sum((consistency_residual - residual_mean) ** 2)
                gd_loss_batch.append(gd_loss_sample)

            gd_loss = torch.mean(torch.stack(gd_loss_batch))
        
        elif self.loss_type == 'optimized':
            
            # 1. Calculate Total Excess Gibbs Energy of the Mixture (g^E)
            # Formula: g^E = sum(x_i * g_i^E)  (Source 79, Eq B14)
            weighted_energy = mole_frac * g_excess_partial
            g_excess_total = scatter_add(weighted_energy, component_batch, dim=0)
            
            # 2. Calculate Gradient of Total Energy w.r.t. Mole Fractions
            # Formula: ∂(g^E) / ∂(x_j)
            (gradient_total_energy,) = autograd.grad(
                outputs=torch.sum(g_excess_total),
                inputs=mole_frac,
                retain_graph=True,
                create_graph=self.track_graph
            )
            
            # 3. Calculate Consistency Residual (v_j)
            # Derived Relation: v_j = ∂g^E/∂x_j - g_j^E  (Source 97, Eq B11)
            consistency_residual = gradient_total_energy - g_excess_partial
            
            # 4. Enforce GD consistency: Variance of v_j must be 0
            residual_mean_per_mixture = scatter_mean(consistency_residual, component_batch, dim=0)
            residual_mean_expanded = residual_mean_per_mixture[component_batch]
            variance_penalty = (consistency_residual - residual_mean_expanded) ** 2
            gd_loss_batch = scatter_add(variance_penalty, component_batch, dim=0)
            
            gd_loss = torch.mean(gd_loss_batch)

        return gd_loss