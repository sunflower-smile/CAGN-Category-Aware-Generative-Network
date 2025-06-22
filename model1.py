import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Utility ===
def kl_divergence(mu_q, logvar_q, mu_p, logvar_p=None):
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    # KL(q||p) per time-step and batch, then mean over all
    kl = 0.5 * (
        (var_q / var_p)
        + ((mu_q - mu_p) ** 2) / var_p
        + logvar_p
        - logvar_q
        - 1
    )
    return kl.sum(dim=-1).mean()

# === Joint Encoder (BiLSTM + Variational) ===
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_layers, dropout, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        h, _ = self.lstm(x)             # h: (B, T, 2*hidden_dim)
        h = self.norm(h)
        mu = self.fc_mu(h)              # (B, T, latent_dim)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps              # reparameterization
        return z, mu, logvar, h

# === AR + Class Label Prior with PoE ===
class ARCategoryPriorPoE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes, rnn_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = rnn_layers
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        # learnable initial hidden and cell states
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.class_mu = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.class_logvar = nn.Parameter(torch.zeros(num_classes, latent_dim))

    def forward(self, z, y_onehot):
        # z: (B, T, L)
        B, T, _ = z.size()
        # prepare learnable initial states per batch
        h0 = self.h0.expand(-1, B, -1).contiguous()
        c0 = self.c0.expand(-1, B, -1).contiguous()
        # autoregressive shift
        z_shift = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        h, _ = self.lstm(z_shift, (h0, c0))
        h = self.norm(h)
        mu_ar = self.to_mu(h)
        logvar_ar = torch.clamp(self.to_logvar(h), min=-10, max=10)
        # class prior
        mu_cls = torch.matmul(y_onehot, self.class_mu)               # (B, latent_dim)
        mu_cls = mu_cls.unsqueeze(1).expand(-1, T, -1)               # (B, T, latent_dim)
        logvar_cls = torch.clamp(
            torch.matmul(y_onehot, self.class_logvar).unsqueeze(1).expand(-1, T, -1),
            min=-10,
            max=10
        )
        # Product of Experts
        var_ar = torch.exp(logvar_ar)
        var_cls = torch.exp(logvar_cls)
        var_poe = 1.0 / (1.0 / var_ar + 1.0 / var_cls)
        mu_poe = var_poe * (mu_ar / var_ar + mu_cls / var_cls)
        logvar_poe = torch.log(var_poe)
        return mu_poe, logvar_poe, mu_ar, logvar_ar, mu_cls, logvar_cls

# === GM Decoder with Full Covariance ===
class Decoder(nn.Module):
    def __init__(self, latent_dim, data_dim, num_components):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.num_components = num_components
        self.to_logits = nn.Linear(latent_dim, num_components)
        # component means: (K, D)
        self.means = nn.Parameter(torch.randn(num_components, data_dim))
        # unconstrained lower-triangular factors for covariance
        L_init = torch.stack([torch.eye(data_dim) for _ in range(num_components)], dim=0)
        self.cov_L = nn.Parameter(L_init)

    def forward(self, z, tau=1.0):
        # z: (B, T, L)
        B, T, _ = z.shape
        K, D = self.num_components, self.data_dim
        # mixture weights
        logits = self.to_logits(z)                     # (B, T, K)
        weights = F.softmax(logits / tau, dim=-1)      # (B, T, K)
        # mixture means
        mean = torch.einsum("btk,kd->btd", weights, self.means)
        # reconstruct full covariance matrices
        L_un = self.cov_L                              # (K, D, D)
        tril_indices = torch.tril_indices(D, D, offset=-1)
        diag_indices = torch.arange(D)
        L = torch.zeros_like(L_un)
        # lower off-diagonal
        L[:, tril_indices[0], tril_indices[1]] = L_un[:, tril_indices[0], tril_indices[1]]
        # diagonal (positive)
        L[:, diag_indices, diag_indices] = F.softplus(L_un[:, diag_indices, diag_indices])
        # Sigma_k = L @ L^T
        Sigma = torch.matmul(L, L.transpose(-1, -2))   # (K, D, D)
        # mixture covariance: cov_{b,t} = sum_k w_{b,t,k} * Sigma_k
        cov = torch.einsum("btk,kij->btij", weights, Sigma)  # (B, T, D, D)
        return mean, cov

# === Temporal Attention Pooling ===
class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, h):
        # h: (B, T, D)
        scores = self.attn(h)                  # (B, T, 1)
        weights = F.softmax(scores, dim=1)     # (B, T, 1)
        return (h * weights).sum(dim=1)        # (B, D)

# === TCN Classifier ===
class TCNClassifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):  # z: (B, T, D)
        z = z.transpose(1, 2)                # -> (B, D, T)
        features = self.tcn(z)
        return self.classifier(features)

# === Full Model ===
class LAVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            config['data_dim'],
            config['rnn_hidden'],
            config['rnn_layers'],
            config['dropout'],
            config['latent_dim']
        )
        self.prior = ARCategoryPriorPoE(
            config['latent_dim'],
            config['rnn_hidden'],
            config['num_classes'],
            rnn_layers=config['rnn_layers']
        )
        self.decoder = Decoder(
            config['latent_dim'],
            config['data_dim'],
            config['num_components']
        )
        self.classifier = TCNClassifier(
            config['latent_dim'],
            config['cls_hidden'],
            config['num_classes']
        )
        self.skip_proj = nn.Linear(config['rnn_hidden'] * 2, config['latent_dim'])

    def forward(self, x, y, tau=1.0):
        # prepare one-hot labels
        y_onehot = F.one_hot(y, num_classes=self.config['num_classes']).float()
        # encode
        z, mu_q, logvar_q, h = self.encoder(x)
        # skip connection
        h_skip = self.skip_proj(h)
        z = z + h_skip
        # get prior parameters
        mu_poe, logvar_poe, mu_ar, logvar_ar, mu_cls, logvar_cls = self.prior(z, y_onehot)
        # KL loss
        kl_loss = kl_divergence(mu_q, logvar_q, mu_poe, logvar_poe)
        # decode (full covariance)
        means, cov = self.decoder(z, tau=tau)
        # reconstruction log-likelihood
        B, T, D = means.shape
        diff = x - means                           # (B, T, D)
        # invert covariance per sample-time
        cov_flat = cov.view(B * T, D, D)
        inv_cov = torch.inverse(cov_flat)
        inv_cov = inv_cov.view(B, T, D, D)
        # Mahalanobis term
        mah = torch.einsum('btd,btij,btj->bt', diff, inv_cov, diff)
        # log-determinant term
        sign, logdet = torch.slogdet(cov_flat)
        logdet = logdet.view(B, T)
        const = D * math.log(2 * math.pi)
        recon_ll = -0.5 * (mah + logdet + const)
        recon_loss = -recon_ll.mean() 
        # classification loss
        logits = self.classifier(z)
        cls_loss = F.cross_entropy(logits, y)
        return {
            'logits': logits,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'cls_loss': cls_loss,
            'z': z,
            'mu_q': mu_q,
            'logvar_q': logvar_q,
            'mu_ar': mu_ar,
            'logvar_ar': logvar_ar,
            'mu_cls': mu_cls,
            'logvar_cls': logvar_cls,
            'mu_poe': mu_poe,
            'logvar_poe': logvar_poe,
        }