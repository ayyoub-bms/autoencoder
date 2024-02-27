import torch
from torch import nn

from research.pytorch.utils import RegularizedModule


class LinearAutoEncoder(nn.Module):

    def __init__(self, nb_managed_ptfs, nb_fctr):
        super(FactorAETest, self).__init__()

        self.encoder = nn.Linear(in_features=nb_managed_ptfs,
                                 out_features=nb_fctr)

        self.decoder = nn.Linear(in_features=nb_fctr,
                                 out_features=nb_managed_ptfs)

    def forward(self, sample):
        factors = self.encoder(sample)
        return self.decoder(factors), factors


class FactorBase(RegularizedModule):
    def __init__(self):
        super(FactorBase, self).__init__(use_bn)
        self._factors = None
        self._loadings = None
        self._use_bn = use_bn

    @property
    def factors(self):
        return self._factors

    @property
    def loadings(self):
        return self._loadings

    def _predict(self):
        return torch.squeeze(
            torch.matmul(self._loadings, self._factors.permute(0, 2, 1))
        )


class AE0(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr, use_bn=True):
        super(AE0, self).__init__(use_bn)

        self.beta_l1 = nn.Linear(in_features=nb_char,
                                 out_features=nb_fctr)

        self.encoder = nn.Linear(in_features=nb_ptfs,
                                 out_features=nb_fctr)

    def forward(self, char, ptfs):
        self._betas = self.beta_l1(char)
        self._factors = self.encoder(ptfs)
        return self._predict()


class AE1(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr, use_bn=True):
        super(AE1, self).__init__(use_bn)

        self.beta_l1 = nn.Linear(in_features=nb_char,
                                 out_features=32)

        self.beta_l2 = nn.Linear(in_features=32,
                                 out_features=nb_fctr)

        self.encoder = nn.Linear(in_features=nb_ptfs,
                                 out_features=nb_fctr)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)

    def forward(self, char, ptfs):
        #
        # First layer config
        betas = self.beta_l1(char)
        if self._use_bn:
            betas = self.bn1(betas.permute(0, 2, 1)).permute(0, 2, 1)
        betas = self.relu(betas)

        # Output layer config
        self._betas = self.beta_l2(betas)
        self._factors = self.encoder(ptfs)
        return self._predict()


class AE2(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr, use_bn=True):
        super(AE2, self).__init__(use_bn)

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

    def forward(self, char, ptfs, **kwargs):
        # First layer config
        betas = self.beta_l1(char)
        if self._use_bn:
            betas = self.bn1(betas.permute(0, 2, 1)).permute(0, 2, 1)

        betas = self.relu(betas)

        # Second layer config
        betas = self.beta_l2(betas)
        if self._use_bn:
            betas = self.bn2(betas.permute(0, 2, 1)).permute(0, 2, 1)
        betas = self.relu(betas)

        # Output layer config
        self._betas = self.beta_l3(betas)
        self._factors = self.encoder(ptfs)
        return self._predict()


class AE3(FactorBase):

    def __init__(self, nb_char, nb_ptfs, nb_fctr, use_bn=True):
        super(AE3, self).__init__(use_bn)

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

    def forward(self, char, ptfs, **kwargs):

        # First Layer config
        betas = self.beta_l1(char)
        if self._use_bn:
            betas = self.bn1(betas.permute(0, 2, 1)).permute(0, 2, 1)
        betas = self.relu(betas)

        # Second Layer config
        betas = self.beta_l2(betas)
        if self._use_bn:
            betas = self.bn2(betas.permute(0, 2, 1)).permute(0, 2, 1)
        betas = self.relu(betas)

        # Third Layer config
        betas = self.beta_l3(betas)
        if self._use_bn:
            betas = self.bn3(betas.permute(0, 2, 1)).permute(0, 2, 1)
        betas = self.relu(betas)

        # Output layer config
        self._betas = self.beta_l4(betas)
        self._factors = self.encoder(ptfs)
        return self._predict()
