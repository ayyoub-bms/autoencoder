import torch
from torch import nn


class RegularizedModule(torch.nn.Module):
    """Class that implements regularizations of a neural network
    """

    def __init__(self):
        super(RegularizedModule, self).__init__()

    def _get_params(self, bias=False):
        params = []
        if bias:
            for param in self.parameters():
                params.append(param.view(-1))
            return torch.cat(params)

        for name, param in self.named_parameters():
            if 'bias' not in name:
                params.append(param.view(-1))
        return torch.cat(params)

    def lasso(self, lasso_param):
        params = self._get_params()
        return lasso_param * torch.linalg.norm(params, 1)

    def ridge(self, ridge_param):
        params = self._get_params()
        return ridge_param * torch.linalg.norm(params, 2)

    def elastic_net(self, lasso_param, ridge_param):
        return self.ridge(ridge_param) + self.lasso(lasso_param)


class LinearAutoEncoder(nn.Module):

    def __init__(self, nb_managed_ptfs, nb_fctr):
        super(LinearAutoEncoder, self).__init__()

        self.encoder = nn.Linear(in_features=nb_managed_ptfs,
                                 out_features=nb_fctr)

        self.decoder = nn.Linear(in_features=nb_fctr,
                                 out_features=nb_managed_ptfs)

    def forward(self, sample):
        factors = self.encoder(sample)
        return self.decoder(factors), factors


class FactorBase(RegularizedModule):
    def __init__(self, nb_factors):
        super(FactorBase, self).__init__()
        self._nb_factors = nb_factors
        self._factors = None
        self._loadings = None

    @property
    def factors(self):
        return self._factors

    @property
    def loadings(self):
        return self._loadings

    def _predict(self):
        return torch.squeeze(self._loadings @ self._factors)


class AE0(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr):
        super(AE0, self).__init__(nb_factors=nb_fctr)

        self.beta_l1 = nn.Linear(in_features=nb_char,
                                 out_features=nb_fctr)

        self.encoder = nn.Linear(in_features=nb_ptfs,
                                 out_features=nb_fctr)

        self.dropout = nn.Dropout(.2)

    def forward(self, char, ptfs):
        T, N, Pc = char.shape
        char = self.dropout(char.view(T*N, Pc))
        self._loadings = self.beta_l1(char).view(T, N, self._nb_factors)
        ptfs = self.dropout(ptfs)
        self._factors = self.encoder(ptfs).view(T, self._nb_factors, 1)
        return self._predict()


class AE1(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr):
        super(AE1, self).__init__(nb_factors=nb_fctr)

        self.beta_l1 = nn.Linear(in_features=nb_char,
                                 out_features=32)

        self.beta_l2 = nn.Linear(in_features=32,
                                 out_features=nb_fctr)

        self.encoder = nn.Linear(in_features=nb_ptfs,
                                 out_features=nb_fctr)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(.2)

    def forward(self, char, ptfs):
        # First layer config
        T, N, Pc = char.shape
        char = char.view(T*N, Pc)
        betas = self.beta_l1(self.dropout(char))
        betas = self.bn1(betas)
        betas = self.relu(betas)

        # Output layer config
        self._loadings = self.beta_l2(betas).view(T, N, self._nb_factors)
        ptfs = self.dropout(ptfs)
        self._factors = self.encoder(ptfs).view(T, self._nb_factors, 1)
        return self._predict()


class AE2(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr):
        super(AE2, self).__init__(nb_factors=nb_fctr)

        self.beta_l1 = nn.Linear(in_features=nb_char,
                                 out_features=32)

        self.beta_l2 = nn.Linear(in_features=32,
                                 out_features=16)

        self.beta_l3 = nn.Linear(in_features=16,
                                 out_features=nb_fctr)

        self.encoder = nn.Linear(in_features=nb_ptfs,
                                 out_features=nb_fctr)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(.2)

    def forward(self, char, ptfs, **kwargs):
        # First layer config
        T, N, Pc = char.shape
        char = char.view(T*N, Pc)
        char = self.dropout(char)
        betas = self.beta_l1(char)
        betas = self.bn1(betas)
        betas = self.relu(betas)

        # Second layer config
        betas = self.dropout(betas)
        betas = self.beta_l2(betas)
        betas = self.bn2(betas)
        betas = self.relu(betas)

        # Output layer config
        self._loadings = self.beta_l3(betas).view(T, N, self._nb_factors)
        ptfs = self.dropout(ptfs)
        self._factors = self.encoder(ptfs).view(T, self._nb_factors, 1)
        return self._predict()


class AE3(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr):
        super(AE3, self).__init__(nb_factors=nb_fctr)

        self.beta_l1 = nn.Linear(in_features=nb_char,
                                 out_features=32)

        self.beta_l2 = nn.Linear(in_features=32,
                                 out_features=16)

        self.beta_l3 = nn.Linear(in_features=16,
                                 out_features=8)

        self.beta_l4 = nn.Linear(in_features=8,
                                 out_features=nb_fctr)

        self.encoder = nn.Linear(in_features=nb_ptfs,
                                 out_features=nb_fctr)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(.2)

    def forward(self, char, ptfs, **kwargs):

        T, N, Pc = char.shape
        char = char.view(T*N, Pc)
        char = self.dropout(char)
        # First Layer config
        betas = self.beta_l1(char)
        betas = self.bn1(betas)
        betas = self.relu(betas)

        # Second Layer config
        betas = self.dropout(betas)
        betas = self.beta_l2(betas)
        betas = self.bn2(betas)
        betas = self.relu(betas)

        # Third Layer config
        betas = self.dropout(betas)
        betas = self.beta_l3(betas)
        betas = self.bn3(betas)
        betas = self.relu(betas)

        # Output layer config
        self._loadings = self.beta_l4(betas).view(T, N, self._nb_factors)
        ptfs = self.dropout(ptfs)
        self._factors = self.encoder(ptfs).view(T, self._nb_factors, 1)
        return self._predict()
