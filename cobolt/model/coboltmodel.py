# code reference:
# https://github.com/AntixK/PyTorch-VAE
# https://github.com/mhw32/multimodal-vae-public

import itertools
import torch
from torch import nn
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from torch import logsumexp
from torch.distributions import Normal


def xavier_init(fan_in, fan_out, constant=1):
    """
    Xavier Initialization

    return
    ----------
    weight initialization with dimensions [fan_in, fan_out]
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return (low - high) * torch.rand(fan_in, fan_out) + high


class CoboltModel(nn.Module):

    def __init__(self,
                 in_channels: List,
                 omic: List,
                 latent_dim: int,
                 n_dataset: List,
                 hidden_dims: List = None,
                 alpha: float = None,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 intercept_adj: bool = True,
                 slope_adj: bool = True,
                 log: bool = True,
                 elbo_combn: list = None):
        super(CoboltModel, self).__init__()

        self.latent_dim = latent_dim
        if alpha is None:
            self.alpha = 50.0 / latent_dim
        else:
            self.alpha = alpha
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.experts = ProductOfExperts()
        self.n_dataset = n_dataset
        self.intercept_adj = intercept_adj
        self.slope_adj = slope_adj
        self.log = log
        self.omic = omic
        """
        needs omic because want to know which is methy 
        also remember to __init__ omic and input omic when calling coboltmodel
        """

        self.beta = nn.ParameterList()
        for in_ch in in_channels:
            self.beta.append(torch.nn.Parameter(xavier_init(latent_dim, in_ch), requires_grad=True))
        self.beta_dataset = nn.ParameterList()
        for in_ch, n_d in zip(in_channels, n_dataset):
            self.beta_dataset.append(torch.nn.Parameter(xavier_init(n_d, in_ch), requires_grad=True))
        self.beta_dataset_mtp = nn.ParameterList()
        for in_ch, n_d in zip(in_channels, n_dataset):
            self.beta_dataset_mtp.append(torch.nn.Parameter(torch.rand(n_d, in_ch), requires_grad=True))

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Constructing Laplace Approximation to Dirichlet Prior
        # The greater the alpha, the higher the mode. That is, the probs will
        # be more centered around (1/latent_dim, ..., 1/latent_dim)
        self.a = self.alpha * torch.ones(1, self.latent_dim)
        self.mu2 = (torch.log(self.a) - torch.mean(torch.log(self.a), 1)).to(device=self.device)
        self.var2 = (((1 / self.a) * (1 - (2.0 / self.latent_dim))) +
                     (1.0 / (self.latent_dim * self.latent_dim)) * torch.sum(1 / self.a, 1)).to(device=self.device)

        self.encoder = nn.ModuleList()
        self.fc_mu = nn.ModuleList()
        self.fc_var = nn.ModuleList()

        omic_bool = [x=="Methy" for x in self.omic]
        in_channels2 =[c_in*2 if o_bool else c_in for o_bool, c_in in zip(omic_bool, in_channels)]
        """
        input channels for the encoder would be 2 times for methylation
        """

        for in_ch in in_channels2:
            # Build Encoder
            modules = []
            current_in = in_ch
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Linear(current_in, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU())
                )
                current_in = h_dim
            self.encoder.append(nn.Sequential(*modules))
            self.fc_mu.append(nn.Linear(hidden_dims[-1], latent_dim))
            self.fc_var.append(nn.Linear(hidden_dims[-1], latent_dim))

    def reparameterize(self, mu, var):
        # std = torch.exp(0.5 * log_var)
        std = var.sqrt()
        # torch.randn_like
        # Returns a tensor with the same size as input that is filled with
        # random numbers from a normal distribution with mean 0 and variance 1.
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x: List, cov: List, ifmethy: List):
            x1 = list(x)
            batch_size = [x_i.size(0) for x_i in x1 if x_i is not None][0]  ### real batch sizes from the DataLoader
            qz_m, qz_logv = prior_expert(self.mu2, self.var2, batch_size)
            qz_m = qz_m.to(self.device)
            qz_logv = qz_logv.to(self.device)
            mu = [qz_m]
            log_var = [qz_logv]
            t = 0
            for x_i, cov_i in zip(x1, cov):
                if x_i is not None:
                    if cov_i is not None:
                        #   comb = torch.cat((x_i[0], cov_i[0]))
                        comb = []
                        for xi, covi in zip(x_i, cov_i):
                            combi = torch.cat((xi, covi))
                            comb += [combi]
                        #      comb = torch.cat((comb, combi), dim = -1)
                        comb = torch.stack(comb)
                        x1[t] = comb
                t += 1
            for x_i, encoder, fc_mu, fc_var in zip(x1, self.encoder, self.fc_mu, self.fc_var):
                if x_i is not None:
                    if self.log:
                        x_i = torch.log(x_i + 1)  ### methylation should not take log?????? since there are cov and mc
                    result = encoder(x_i)
                    mu += [fc_mu(result).unsqueeze(0)]
                    log_var += [fc_var(result).unsqueeze(0)]
                else:
                    mu += [qz_m]  # this is a placeholder
                    log_var += [qz_logv]  # won't be used b.c. elbo_combn
            mu = torch.cat(mu, dim=0)
            log_var = torch.cat(log_var, dim=0)
            return mu, log_var

    def encode1(self, x: List, cov: List, ifmethy: List):
        x1 = list(x)
        batch_size = [x_i.size(0) for x_i in x1 if x_i is not None][0]  ### real batch sizes from the DataLoader
        qz_m, qz_logv = prior_expert(self.mu2, self.var2, batch_size)
        qz_m = qz_m.to(self.device)
        qz_logv = qz_logv.to(self.device)
        mu = [qz_m]
        log_var = [qz_logv]
        t = 0
        for x_i, cov_i in zip(x1, cov):
            if cov_i is not None:
                #   comb = torch.cat((x_i[0], cov_i[0]))
                comb = []
                for xi, covi in zip(x_i, cov_i):
                    combi = torch.cat((xi, covi))
                    comb += [combi]
                #      comb = torch.cat((comb, combi), dim = -1)
                comb = torch.stack(comb)
                x1[t] = comb
            t += 1
        for x_i, encoder, fc_mu, fc_var in zip(x1, self.encoder, self.fc_mu, self.fc_var):
            if x_i is not None:
                if self.log:
                    x_i = torch.log(x_i + 1)  ### methylation should not take log?????? since there are cov and mc
                result = encoder(x_i)
                mu += [fc_mu(result).unsqueeze(0)]
                log_var += [fc_var(result).unsqueeze(0)]
            else:
                mu += [qz_m]  # this is a placeholder
                log_var += [qz_logv]  # won't be used b.c. elbo_combn
        mu = torch.cat(mu, dim=0)
        log_var = torch.cat(log_var, dim=0)
        return mu, log_var
    def encode2(self, x: List, cov: List, ifmethy: List):
        batch_size = [x_i.size(0) for x_i in x if x_i is not None][0]
        qz_m, qz_logv = prior_expert(self.mu2, self.var2, batch_size)
        qz_m = qz_m.to(self.device)
        qz_logv = qz_logv.to(self.device)
        mu = [qz_m]
        log_var = [qz_logv]
        for x_i, encoder, fc_mu, fc_var in zip(x, self.encoder, self.fc_mu, self.fc_var):
            if x_i is not None:
                if self.log:
                    x_i = torch.log(x_i + 1)
                result = encoder(x_i)
                mu += [fc_mu(result).unsqueeze(0)]
                log_var += [fc_var(result).unsqueeze(0)]
            else:
                mu += [qz_m]  # this is a placeholder
                log_var += [qz_logv]  # won't be used b.c. elbo_combn
        mu = torch.cat(mu, dim=0)
        log_var = torch.cat(log_var, dim=0)
        return mu, log_var

    def forward(self, x: List, elbo_combn=None):
        x, dataset, cov, ifmethy = x ###
        n_modality = len(x)
        if elbo_combn is None:
            elbo_combn = \
                [list(i)
                 for i in itertools.product([False, True], repeat=n_modality)
                 if sum(i) != 0]

        mu, log_var = self.encode(x, cov, ifmethy)  ###TODO: adding cov to encoder if methy
        recon_loss = 0
        latent_loss = 0
        for elbo_bool in elbo_combn: ## summing up loss over different combination of modalities
            mu_subset, var = self.experts(mu[[True] + elbo_bool],
                                          log_var[[True] + elbo_bool])
            z = self.reparameterize(mu_subset, var)
            beta_subset = [self.beta[i] for i, j in enumerate(elbo_bool) if j]
            x_subset = [x[i] for i, j in enumerate(elbo_bool) if j]
            beta_dataset_subset = [self.beta_dataset[i] for i, j in enumerate(elbo_bool) if j]
            beta_dataset_mtp_subset = [self.beta_dataset_mtp[i] for i, j in enumerate(elbo_bool) if j]
            dataset_subset = [dataset[i] for i, j in enumerate(elbo_bool) if j]
            n_fac = [self.n_dataset[i] for i, j in enumerate(elbo_bool) if j]
            cov_subset = [i if j else None for i,j in enumerate(ifmethy) ] ###
            cov_subset = [cov[co] if co is not None else None for co, j in zip(cov_subset, elbo_bool) if j] ###
            ## summing up loss over modalities
            for beta_i, x_i,  beta_dt_i, beta_dt_mtp_i, dt_i, n_f, cov_i in \
            zip(beta_subset, x_subset, beta_dataset_subset, beta_dataset_mtp_subset, dataset_subset, n_fac, cov_subset): ###
                slope_adj = torch.matmul(fac_to_mat(dt_i, n_f), beta_dt_mtp_i) if self.intercept_adj else 0 ### BUG
                intercept_adj = torch.matmul(fac_to_mat(dt_i, n_f), beta_dt_i) if self.slope_adj else 0 ### BUG
                if cov_i is None: ###
                  recon_loss += self.recon_loss_count(x_i, beta_i, slope_adj, intercept_adj, z) ###
                else:
                  recon_loss += self.recon_loss_binom(x_i, cov_i, beta_i, slope_adj, intercept_adj, z) ###
            latent_loss += self.latent_loss(mu_subset, var, self.mu2, self.var2)
        return latent_loss, recon_loss
    def forward2(self, x: List, elbo_combn=None):
        x, dataset, cov, ifmethy = x ##
        n_modality = len(x)
        if elbo_combn is None:
            elbo_combn = \
                [list(i)
                 for i in itertools.product([False, True], repeat=n_modality)
                 if sum(i) != 0]

        mu, log_var = self.encode(x, cov, ifmethy)
        recon_loss = 0
        latent_loss = 0
        for elbo_bool in elbo_combn:
            mu_subset, var = self.experts(mu[[True] + elbo_bool],
                                          log_var[[True] + elbo_bool])
            z = self.reparameterize(mu_subset, var)
            beta_subset = [self.beta[i] for i, j in enumerate(elbo_bool) if j]
            x_subset = [x[i] for i, j in enumerate(elbo_bool) if j]
            beta_dataset_subset = [self.beta_dataset[i] for i, j in enumerate(elbo_bool) if j]
            beta_dataset_mtp_subset = [self.beta_dataset_mtp[i] for i, j in enumerate(elbo_bool) if j]
            dataset_subset = [dataset[i] for i, j in enumerate(elbo_bool) if j]
            n_fac = [self.n_dataset[i] for i, j in enumerate(elbo_bool) if j]
            for beta_i, x_i, beta_dt_i, beta_dt_mtp_i, dt_i, n_f in \
                    zip(beta_subset, x_subset, beta_dataset_subset, beta_dataset_mtp_subset, dataset_subset, n_fac):
                slope_adj = torch.matmul(fac_to_mat(dt_i, n_f), beta_dt_mtp_i) if self.intercept_adj else 0
                intercept_adj = torch.matmul(fac_to_mat(dt_i, n_f), beta_dt_i) if self.slope_adj else 0
                recon_loss += - torch.sum(
                    x_i * torch.log(torch.softmax(
                        torch.matmul(torch.softmax(z, dim=1), beta_i) * (slope_adj + 1) + intercept_adj,
                        dim=1)))
            latent_loss += self.latent_loss(mu_subset, var, self.mu2, self.var2)

        return latent_loss, recon_loss

    def latent_loss(self, mu0, var0, mu1, var1):
        latent_loss = 0.5 * torch.sum(  # minimize risk, maximize ELBO
            var0 / var1 + (mu1 - mu0) / var1 * (mu1 - mu0) - self.latent_dim + \
            torch.log(var1) - torch.log(var0))
        return latent_loss

    def recon_loss_binom(self, x, cov, beta, slope, intercept, latent):
        logitp = torch.matmul(torch.softmax(latent, dim=1), beta) * (slope + 1) + intercept
        p = torch.exp(logitp)/(1+torch.exp(logitp))
        loss = - torch.sum(x*torch.log(p)+(cov-x)*torch.log(1-p))
        return(loss)
    def recon_loss_count(self, x, beta, slope, intercept, latent):
        loss = - torch.sum( ## x_i and the softmatrix are of same dimensions,
                    x * torch.log(torch.softmax(
                        torch.matmul(torch.softmax(latent, dim=1), beta) * (slope + 1) + intercept,
                        dim=1)))
        return(loss)
    @torch.no_grad()
    def get_beta(self):
        return [beta.cpu().numpy().T for beta in self.beta]

    @torch.no_grad()
    def get_topic_prop(self, x, elbo_bool=None):
        x, dataset, cov, ifmethy = x
        # TODO: This is not the posterior mean of \theta
        mu, log_var = self.encode(x, cov, ifmethy)
        if elbo_bool is None:
            elbo_bool = [True]*len(x)
        mu_subset, var = self.experts(mu[[True] + elbo_bool],
                                      log_var[[True] + elbo_bool])
        return torch.softmax(mu_subset, dim=1).cpu().numpy()

    @torch.no_grad()
    def get_latent(self, x, elbo_bool=None):
        x, dataset, cov, ifmethy = x
        mu, log_var = self.encode(x, cov, ifmethy)
        if elbo_bool is None:
            elbo_bool = [True]*len(x)
        mu_subset, var = self.experts(mu[[True] + elbo_bool],
                                      log_var[[True] + elbo_bool])
        return mu_subset.cpu().numpy()

    @torch.no_grad()
    def get_posterior(self, x, cov, ifmethy, elbo_bool=None):
        mu, log_var = self.encode(x, cov, ifmethy)
        if elbo_bool is None:
            elbo_bool = [True]*len(x)
        mu_subset, var = self.experts(mu[[True] + elbo_bool],
                                      log_var[[True] + elbo_bool])
        return mu_subset, var

    @torch.no_grad()
    def get_marginal_likelihood(self, x, cov, ifmethy, elbo_bool=None, rep=100):
        """
        Reference scVI
        """
        if elbo_bool is None:
            elbo_bool = [True]*len(x)

        mu, log_var = self.encode(x, cov, ifmethy)
        mu_subset, var = self.experts(mu[[True] + elbo_bool],
                                      log_var[[True] + elbo_bool])

        beta_subset = [self.beta[i] for i, j in enumerate(elbo_bool) if j]
        x_subset = [x[i] for i, j in enumerate(elbo_bool) if j]
        l_sum = torch.zeros(x[0].size()[0], rep)
        for i in range(rep):
            z = self.reparameterize(mu_subset, var)
            z1 = self.reparameterize(mu_subset, var)
            recon_loss = torch.zeros(x[0].size()[0]).to(device=self.device)
            for beta_i, x_i in zip(beta_subset, x_subset):
                recon_loss += - torch.sum(
                    x_i * torch.log(torch.softmax(
                        torch.matmul(torch.softmax(z1, dim=1), beta_i),
                        dim=1)), dim=1)
            p_z = Normal(torch.zeros_like(self.mu2), torch.ones_like(self.var2).sqrt()).log_prob(z).sum(dim=-1)
            q_z_x = Normal(mu_subset, var.sqrt()).log_prob(z).sum(dim=-1)
            p_x_z = - recon_loss
            l_sum[:, i] += (p_z + p_x_z - q_z_x).cpu().numpy()
        batch_log_lkl = logsumexp(l_sum, dim=-1) - np.log(rep)
        return - torch.sum(batch_log_lkl).item()

    @torch.no_grad()
    def plot_beta(self, path=None):
        dt = self.get_beta()
        for i, x in enumerate(dt):
            fig = plt.figure()
            sns.heatmap(x, cmap="Blues")
            if path is not None:
                fig.savefig(path + 'beta' + str(i) + '.png')
                plt.close()


class ProductOfExperts(nn.Module):
    def forward(self, mu, log_var):
        """Return parameters for product of independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.

        @param mu: M x D for M experts
        @param log_var: M x D for M experts
        """
        # precision of i-th Gaussian expert at point x
        var = torch.exp(log_var)
        T = 1. / var  # the variance matrices are diagonal
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        return pd_mu, pd_var


def prior_expert(mu, var, batch_size):
    mu = mu.repeat((batch_size, 1)).unsqueeze(0)
    log_var = torch.log(var.repeat((batch_size, 1)).unsqueeze(0))
    return mu, log_var


def fac_to_mat(fac, n_f):
    return torch.stack([(fac == i).float() for i in range(n_f)]).T
