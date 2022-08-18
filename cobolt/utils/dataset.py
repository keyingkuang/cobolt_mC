
import numpy as np
from scipy import sparse
import random
import itertools
import torch
from torch.utils.data import DataLoader
from cobolt.utils.data import MultiData


class MultiomicDataset(torch.utils.data.Dataset):
    def __init__(self, dt: MultiData):
        self.dt = dt.get_data()
        self.omic = list(self.dt.keys())
        self.barcode = self._get_unique_barcode()
        self.dataset = self._get_dataset()
        self.n_dataset = [np.unique(self.dt[om]['dataset']).shape[0] for om in self.omic]
        b_dict = {om: {b: i for i, b in enumerate(self.dt[om]['barcode'])} for om in self.omic}
        self.barcode_in_om = {om: {b: (b_dict[om][b] if b in b_dict[om] else None) for b in self.barcode}
                              for om in self.omic}

    def __len__(self):
        """Number of samples in the data"""
        return self.barcode.shape[0]

    def __getitem__(self, index: int):
        """
        Generates one sample of data
        ----------------------------
        dat : a list storing the counts for each of the omic
        dataset: a list storing the respective dataset code
        ifcov: a list storing TRUE/FALSE values indicating if/not this omic is contained in the cells selected AND this omic is methylation
        cov: a list storing the coverages for each of the omic
        """

        b = self.barcode[index]
        dat = [self.dt[om]['counts'][self.barcode_in_om[om][b]] if self.barcode_in_om[om][b] is not None else None
               for om in self.omic]
        dataset = [self.dt[om]['dataset'][self.barcode_in_om[om][b]] if self.barcode_in_om[om][b] is not None else None
               for om in self.omic]
        ifcov = [x=="Methy" for x in self.omic]
        datasettf = [x is not None for x in dataset]
        ifcov = [x&dat for x, dat in zip(ifcov, datasettf)]
        cov = [self.dt[om]['coverage'][self.barcode_in_om[om][b]] if ic else None ###
               for om,ic in zip(self.omic, ifcov)]
        return dat, dataset, cov, ifcov

    def __str__(self):
        n_modality = len(self.omic)
        s1 = "A MultiomicDataset object with {} omics:\n".format(n_modality)
        s2 = "".join(["- {}: {} features, {} cells, {} batches.\n".format(
            om, len(self.dt[om]['feature']), len(self.dt[om]['barcode']), self.n_dataset[i])
            for i, om in enumerate(self.omic)])
        s3 = "Joint cells:\n"
        joint_omic = [list(i) for i in itertools.product([False, True], repeat=n_modality) if sum(i) > 1]
        s4 = "\n".join(["- {}: {} cells.".format(
            ", ".join([om for i, om in enumerate(self.omic) if om_combn[i]]),
            len(self.get_comb_idx(om_combn)))
            for om_combn in joint_omic])
        return s1 + s2 + s3 + s4

    @classmethod
    def from_singledata(cls, *single_data):
        return cls(MultiData(*single_data))

    def _get_unique_barcode(self):
        barcode = np.concatenate([self.dt[om]['barcode'] for om in self.omic])
        return np.unique(barcode)

    def _get_dataset(self):
        dt_dict = {}
        for om in self.omic:
            dataset_names = [self.dt[om]['dataset_name'][int(i)] for i in self.dt[om]['dataset']]
            for b, d in zip(self.dt[om]['barcode'], dataset_names):
                if b not in dt_dict:
                    dt_dict[b] = d
                elif d != dt_dict[b]:
                    raise ValueError("Duplicate barcode found: {}".format(b))
        return dt_dict

    def get_barcode(self):
        return self.barcode

    def get_comb_idx(self, omic_combn):
        if not any(omic_combn):
            raise ValueError("Omics combination can not be all False.")
        if len(omic_combn) != len(self.get_feature_shape()):
            raise ValueError(
                "omic_combn should be a boolean list of length {}".format(
                    len(self.get_feature_shape())))

        bl = [self.dt[om]['barcode'] for om, include in zip(self.omic, omic_combn) if include]
        b = bl[0]
        for x in bl[1:]:
            b = np.intersect1d(b, x)
        return np.where(np.isin(self.barcode, b))[0]

    def get_feature_shape(self):
        return [self.dt[om]['feature'].shape[0] for om in self.omic]
