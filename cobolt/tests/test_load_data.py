import pytest
import os
import numpy as np
import scipy

# THIS_DIR = "/content/cobolt_mC/cobolt/tests"
"""
need to specify the directory before running the tests
"""

def load_test_data():
    ja = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_a"),
        dataset_name="joint", feature_name="a")
    jb = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_b"),
        dataset_name="joint", feature_name="b")
    sa = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_a"),
        dataset_name="single_a", feature_name="a")
    sb = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_b"),
        dataset_name="single_b", feature_name="b")
    return ja, jb, sa, sb


def load_test_data_with_methy():
    ja = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_a"),
        dataset_name="joint", feature_name="a")
    jb = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_b"),
        dataset_name="joint", feature_name="b")
    jc = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_c"),
        dataset_name="joint", feature_name="Methy")
    sa = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_a"),
        dataset_name="single_a", feature_name="a")
    sb = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_b"),
        dataset_name="single_b", feature_name="b")
    sc = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_c"),
        dataset_name="single_c", feature_name="Methy")
    return ja, jb, jc, sa, sb, sc


class TestSingleData:
    ## TODO: add test for __get_item__ when selecting individual items (is.valid() and removing duplicates)
    def test_construction(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        count, coverage, feature, barcode = ja.get_data()
        assert ja.get_dataset_name() == "joint"
        assert count[feature_name].shape == (100, 100)
        assert feature[feature_name].shape == (100,)  ##
        assert barcode.shape == (100,)
        assert isinstance(feature[feature_name], np.ndarray)
        assert isinstance(barcode, np.ndarray)
        assert isinstance(count[feature_name], scipy.sparse.csr.csr_matrix)

    def test_construction_Methy(self):
        feature_name = "Methy"
        ja, jb, jc, sa, sb, sc = load_test_data_with_methy()
        count, coverage, feature, barcode = jc.get_data()
        assert jc.get_dataset_name() == "joint"
        assert count[feature_name].shape == (100, 100)
        assert coverage[feature_name].shape == (100, 100)
        assert feature[feature_name].shape == (100,)
        assert barcode.shape == (100,)
        assert isinstance(feature[feature_name], np.ndarray)
        assert isinstance(barcode, np.ndarray)
        assert isinstance(count[feature_name], scipy.sparse.csr.csr_matrix)

    def test_filter_features(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        ja.filter_features(min_count=2, min_cell=1)
        count, coverage, feature, barcode = ja.get_data()
        assert count[feature_name].shape == (100, 46)
        assert (count[feature_name].sum(axis=0) > 2).all()
        assert ((count[feature_name] != 0).sum(axis=0) > 1).all()
        assert feature[feature_name].shape == (46,)
        assert barcode.shape == (100,)

    def test_filter_cells(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        ja.filter_cells(min_count=2, min_feature=1)
        count, coverage, feature, barcode = ja.get_data()
        assert count[feature_name].shape == (85, 100)
        assert (count[feature_name].sum(axis=1) > 2).all()
        assert ((count[feature_name] != 0).sum(axis=1) > 1).all()
        assert feature[feature_name].shape == (100,)
        assert barcode.shape == (85,)

    def test_filter_barcode(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        ja.filter_barcode(cells=[
            'joint~09A_CAGCCCCGCCTT',
            'joint~09A_CGCCTACCATGA'
        ])
        count, coverage, feature, barcode = ja.get_data()
        assert count[feature_name].shape == (2, 100)
        assert feature[feature_name].shape == (100,)
        assert barcode.shape == (2,)


class TestMultiData:
    def test_construction(self):
        ja, jb, sa, sb = load_test_data()
        multi = MultiData(ja, jb, sa, sb).get_data()
        assert list(multi.keys()) == ['a', 'b']
        assert len(multi['a']['feature']) == 73
        assert len(multi['a']['barcode']) == 200
        assert len(multi['a']['dataset']) == 200
        assert multi['a']['dataset_name'] == ['joint', 'single_a']
        assert multi['a']['counts'].shape == (200, 73)
        assert sum(multi['a']['dataset'] == 0) == 100
        assert sum(multi['a']['dataset'] == 1) == 100
        assert len(multi['b']['feature']) == 500
        assert len(multi['b']['barcode']) == 200
        assert len(multi['b']['dataset']) == 200
        assert multi['b']['dataset_name'] == ['joint', 'single_b']
        assert multi['b']['counts'].shape == (200, 500)
        assert sum(multi['b']['dataset'] == 0) == 100
        assert sum(multi['b']['dataset'] == 1) == 100

    def test_construction_methy(self):
        ja, jb, jc, sa, sb, sc = load_test_data_with_methy()
        multi = MultiData(ja, jb, jc, sa, sb, sc).get_data()
        assert len(multi) == 3
        assert list(multi.keys()) == ['a', 'b', 'Methy']
        assert len(multi['a']['feature']) == 73
        assert len(multi['a']['barcode']) == 200
        assert len(multi['a']['dataset']) == 200
        assert multi['a']['dataset_name'] == ['joint', 'single_a']
        assert multi['a']['counts'].shape == (200, 73)
        assert sum(multi['a']['dataset'] == 0) == 100
        assert sum(multi['a']['dataset'] == 1) == 100
        assert len(multi['b']['feature']) == 500
        assert len(multi['b']['barcode']) == 200
        assert len(multi['b']['dataset']) == 200
        assert multi['b']['dataset_name'] == ['joint', 'single_b']
        assert multi['b']['counts'].shape == (200, 500)
        assert sum(multi['b']['dataset'] == 0) == 100
        assert sum(multi['b']['dataset'] == 1) == 100
        assert multi['Methy']['dataset_name'] == ['joint', 'single_c']
        assert multi['Methy']['counts'].shape == (110, 10)
        assert sum(multi['Methy']['dataset'] == 0) == 100
        assert sum(multi['Methy']['dataset'] == 1) == 10


class TestDataset:
    def test_construction(self):
        ja, jb, sa, sb = load_test_data()
        multi = MultiomicDataset.from_singledata(ja, jb, sa, sb)
        assert len(multi) == 300
        assert multi.get_feature_shape() == [73, 500]
        assert multi.get_barcode().shape == (300,)
        assert multi.get_comb_idx([True, True]).shape == (100,)
        assert multi.get_comb_idx([True, False]).shape == (200,)
        assert multi.get_comb_idx([False, True]).shape == (200,)
        with pytest.raises(ValueError):
            multi.get_comb_idx([False, False])
        with pytest.raises(ValueError):
            multi.get_comb_idx([False, True, True])

    def test_construction_methy(self):
        ja, jb, jc, sa, sb, sc = load_test_data_with_methy()
        multi = MultiomicDataset.from_singledata(ja, jb, jc, sa, sb, sc)
        assert len(multi) == 310
        assert multi.get_feature_shape() == [73, 500, 10]
        assert multi.get_barcode().shape == (310,)
        assert multi.get_comb_idx([True, True, True]).shape == (100,)
        assert multi.get_comb_idx([True, False, False]).shape == (200,)
        assert multi.get_comb_idx([False, True, False]).shape == (200,)
        assert multi.get_comb_idx([False, True, True]).shape == (100,)
        assert multi.get_comb_idx([False, False, True]).shape == (110,)
        with pytest.raises(ValueError):
            multi.get_comb_idx([False, False, False])
        with pytest.raises(ValueError):
            multi.get_comb_idx([False, True])