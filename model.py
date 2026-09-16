import copy

import torch
import torch.nn as nn
from sklearn.cluster import KMeans

L2norm = nn.functional.normalize


class DIVIDE(torch.nn.Module):
    def __init__(self, layer_dims, temperature, n_classes, drop_rate=0.5,
                 rwr_alpha=0.4, knn_k=10):
        super(DIVIDE, self).__init__()
        self.n_classes = n_classes
        self.rwr_alpha = rwr_alpha
        self.knn_k = knn_k
        total_dim = sum([dims[-1] for dims in layer_dims])
        self.cluster_centers = nn.Parameter(torch.Tensor(n_classes, total_dim))
        self.online_encoder = FCN(layer_dims[0], drop_out=drop_rate)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.projection_lay = MLP(layer_dims[0][-1], layer_dims[0][-1])
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = layer_dims[0][-1]

    def forward(self, data, original_data, momentum=0.99,
                warm_up=False, dynamic_weight=0.5):
        self._update_target_branch(momentum)
        z = [self.online_encoder(data[0]), self.online_encoder(data[1])]
        with torch.no_grad():
            z_t = [self.target_encoder(data[0]), self.target_encoder(data[1])]
        p = [self.projection_lay(z[0]), self.projection_lay(z[1])]

        if warm_up:
            with torch.no_grad():
                z_target = L2norm(self.target_encoder(original_data))
                A1 = self._build_knn_graph(z_target, k=self.knn_k)
                relation = self._build_high_order_graph(
                    A1, alpha=self.rwr_alpha, base_k=self.knn_k)
            z_online = L2norm(self.online_encoder(original_data))
            l_dec = self._compute_dec_loss(z_online, z_target, self.cluster_centers)
        else:
            relation = torch.eye(z[0].shape[0], device=z[0].device)
            l_dec = 0

        l_intra = (self.cl(z[0], z_t[0], relation) +
                   self.cl(z[1], z_t[1], relation)) / 2
        l_inter = (self.cl(p[0], z_t[1], relation) +
                   self.cl(p[1], z_t[0], relation)) / 2
        return l_intra + l_inter + dynamic_weight * l_dec


    @torch.no_grad()
    def _build_knn_graph(self, z, k=20):
        z = L2norm(z)
        N = z.size(0)
        if N <= 1:
            return torch.eye(N, device=z.device)
        k = max(1, min(int(k), N - 1))
        sim = z @ z.t() 
        mask = torch.eye(N, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        topk_sim, indices = torch.topk(sim, k=k, dim=1, largest=True)
        weights = torch.softmax(topk_sim / 0.1, dim=1)
        A = torch.zeros(N, N, device=z.device)
        A.scatter_(1, indices, weights) 
        A = (A + A.t()) / 2
        A = A + torch.eye(N, device=z.device)
        
        return A

    @torch.no_grad()
    def _build_high_order_graph(self, A, alpha=0.15, base_k=20):
        N = A.size(0)
        device = A.device
        
        deg = A.sum(dim=1).clamp(min=1e-8)
        deg_inv_sqrt = deg.pow(-0.5)
        A_norm = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
        
        I = torch.eye(N, device=device)
        A_high = alpha * torch.inverse(I - (1 - alpha) * A_norm)
        
        dynamic_k_high = max(5, int(base_k * 1.5)) 
            
        if dynamic_k_high < N:
            topk_val, topk_ind = torch.topk(A_high, k=dynamic_k_high, dim=1, largest=True)
            
            A_high_sparse = torch.zeros_like(A_high, device=device)
            A_high_sparse.scatter_(1, topk_ind, topk_val)
            A_high = A_high_sparse
            
            A_high = (A_high + A_high.t()) / 2.0

        A_high.fill_diagonal_(1.0)

        return A_high.clamp(min=0.0, max=1e6)

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)
    
    @staticmethod
    def kmeans_clustering(embeddings, n_clusters, n_init=10, seed=None):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        embeddings_np = embeddings.detach().cpu().numpy()
        kmeans.fit(embeddings_np)
        centers = torch.tensor(kmeans.cluster_centers_, device=embeddings.device, dtype=embeddings.dtype)
        return centers

    @staticmethod
    def target_distribution(q):
        frequency = q.sum(dim=0, keepdim=True)
        frequency = frequency / frequency.sum().clamp(min=1e-8)
        p = (q ** 2) / frequency.clamp(min=1e-8)
        return p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _soft_assignment(z, centers, alpha=1.0):
        z_norm_sq = (z ** 2).sum(dim=1, keepdim=True)
        centers_norm_sq = (centers ** 2).sum(dim=1, keepdim=True)
        dist_sq = z_norm_sq + centers_norm_sq.t() - 2 * torch.mm(z, centers.t())
        dist_sq = dist_sq.clamp(min=0)
        q = (1.0 + dist_sq / alpha) ** (-(alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def _compute_dec_loss(self, z_online, z_target, centers, alpha=1.0):
        with torch.no_grad():
            q_t = self._soft_assignment(z_target, centers, alpha)
            p_target = self.target_distribution(q_t)
        q_online = self._soft_assignment(z_online, centers, alpha)
        kl_loss = (p_target * (torch.log(p_target + 1e-8) - torch.log(q_online + 1e-8))).sum(dim=1).mean()
        return kl_loss

    @torch.no_grad()
    def _extract_feature(self, data):
        z = self.target_encoder(data)
        z = L2norm(z)
        return z
    


class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=False):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=nn.ReLU):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden, bias=False),
                                 nn.BatchNorm1d(dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out,bias=True))

    def forward(self, x):
        x = self.mlp(x)
        return x

    
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]  

        device = x_q.device
        if mask_pos is None:
            mask_pos = torch.eye(N, device=device)
        else:
            mask_pos = mask_pos.to(device).float()
        
        logits = torch.mm(x_q, x_k.t()) / self.temperature
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  
        pos_weight = mask_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)  

        loss_per_sample = -(mask_pos * log_prob).sum(dim=1, keepdim=True) / pos_weight  
        nll_loss = loss_per_sample.mean()
        
        if torch.isnan(nll_loss) or torch.isinf(nll_loss):
            raise RuntimeError("Training collapsed due to NaN loss in InfoNCE.")
            
        return nll_loss
