import numpy as np
import torch
import pytest

from rl4evrp.environment.instances import generate_instance, build_node_features, make_dataset


class TestGenerateInstance:
    def test_shape(self):
        inst = generate_instance(n_customers=10, seed=0)
        assert inst["coords"].shape == (11, 2)
        assert inst["demands"].shape == (11,)
        assert inst["node_types"].shape == (11,)

    def test_depot_at_index_zero(self):
        inst = generate_instance(n_customers=10, seed=0)
        assert inst["node_types"][0] == 0

    def test_depot_has_zero_demand(self):
        inst = generate_instance(n_customers=10, seed=0)
        assert inst["demands"][0] == 0.0

    def test_chargers_have_zero_demand(self):
        inst = generate_instance(n_customers=20, seed=0, charger_prob=0.5)
        chargers = inst["node_types"] == 2
        assert (inst["demands"][chargers] == 0).all()

    def test_coords_in_unit_square(self):
        inst = generate_instance(n_customers=15, seed=1)
        assert inst["coords"].min() >= 0.0
        assert inst["coords"].max() <= 1.0

    def test_customer_demands_positive(self):
        inst = generate_instance(n_customers=10, seed=0)
        customers = inst["node_types"] == 1
        assert (inst["demands"][customers] > 0).all()

    def test_reproducibility(self):
        a = generate_instance(n_customers=10, seed=7)
        b = generate_instance(n_customers=10, seed=7)
        np.testing.assert_array_equal(a["coords"], b["coords"])

    def test_different_seeds_differ(self):
        a = generate_instance(n_customers=10, seed=1)
        b = generate_instance(n_customers=10, seed=2)
        assert not np.array_equal(a["coords"], b["coords"])

    def test_n_nodes(self):
        inst = generate_instance(n_customers=5)
        assert inst["n_nodes"] == 6


class TestBuildNodeFeatures:
    def test_shape(self, small_inst):
        feats = build_node_features(small_inst)
        assert feats.shape == (small_inst["n_nodes"], 7)

    def test_is_tensor(self, small_inst):
        feats = build_node_features(small_inst)
        assert isinstance(feats, torch.Tensor)

    def test_depot_flag(self, small_inst):
        feats = build_node_features(small_inst)
        assert feats[0, 4] == 1.0   # is_depot channel
        assert feats[1:, 4].sum() == 0.0

    def test_demand_normalised(self, small_inst):
        feats = build_node_features(small_inst)
        assert feats[:, 2].max() <= 1.0 + 1e-6

    @pytest.fixture
    def small_inst(self):
        return generate_instance(n_customers=8, seed=42)


class TestMakeDataset:
    def test_length(self):
        ds = make_dataset(n=5, n_customers=10)
        assert len(ds) == 5

    def test_each_is_dict(self):
        ds = make_dataset(n=3, n_customers=10)
        assert all(isinstance(inst, dict) for inst in ds)

    def test_unique_instances(self):
        ds = make_dataset(n=3, n_customers=10, seed0=0)
        assert not np.array_equal(ds[0]["coords"], ds[1]["coords"])
