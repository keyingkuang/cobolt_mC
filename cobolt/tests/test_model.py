
import pytest
import os
from cobolt.utils import MultiomicDataset
from cobolt.model import Cobolt
from cobolt.tests.test_load_data import load_test_data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestModel:
    def test_model(self):
        ja, jb, sa, sb = load_test_data()
        multi = MultiomicDataset.from_singledata(ja, jb, sa, sb)
        model = Cobolt(dataset=multi, n_latent=10)
        model.train(num_epochs=1)
        latent, barcode = model.get_all_latent(correction=True)
        assert latent.shape == (300, 10)
        assert barcode.shape == (300,)
        model.clustering()
        clusters = model.get_clusters()
        assert clusters.shape == (300,)
        clusters, bcd = model.get_clusters(return_barcode=True)
        assert (barcode == bcd).all()

    def test_model_methy(self):
        ja, jb, jc, sa, sb, sc = load_test_data_with_methy()
        multi = MultiomicDataset.from_singledata(ja, jb, jc, sa, sb, sc)
        model = Cobolt(dataset=multi, n_latent=10, batch_size=100)
        model.train(num_epochs=2)
        assert model.get_latent([True, False, True]).shape == (100, 10)
        latent, barcode = model.get_all_latent(correction=True)
        assert latent.shape == (310, 10)
        assert barcode.shape == (310,)
        model.clustering()
        clusters = model.get_clusters()
        assert clusters.shape == (310,)
        clusters, bcd = model.get_clusters(return_barcode=True)
        assert (barcode == bcd).all()