"""
Integration tests for multitask datasets.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import numpy as np
import unittest
import pytest
import deepchem as dc


class TestMultitask(unittest.TestCase):
    """
    Sanity tests for multitask data.
    """

    def setUp(self):
        super(TestMultitask, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_multitask_order(self):
        """Test that order of tasks in multitask datasets is preserved."""
        input_file = os.path.join(self.current_dir,
                                  "assets/multitask_example.csv")
        tasks = [
            "task0", "task1", "task2", "task3", "task4", "task5", "task6",
            "task7", "task8", "task9", "task10", "task11", "task12", "task13",
            "task14", "task15", "task16"
        ]

        featurizer = dc.feat.CircularFingerprint(size=1024)

        loader = dc.data.CSVLoader(tasks=tasks,
                                   feature_field="smiles",
                                   featurizer=featurizer)
        dataset = loader.create_dataset(input_file)

        splitter = dc.splits.ScaffoldSplitter()
        train_dataset, test_dataset = splitter.train_test_split(dataset)

        assert np.array_equal(train_dataset.get_task_names(), tasks)
        assert np.array_equal(test_dataset.get_task_names(), tasks)

    def test_multitask_data(self):
        """Test that data associated with a tasks stays associated with it."""
        tasks = ["task0", "task1"]
        n_samples = 100
        n_features = 3
        n_tasks = len(tasks)

        # Generate dummy dataset
        ids = np.array(["C"] * n_samples, dtype=object)
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks))
        w = np.ones((n_samples, n_tasks))

        dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, tasks)
        np.testing.assert_allclose(X, dataset.X)
        np.testing.assert_allclose(y, dataset.y)
        np.testing.assert_allclose(w, dataset.w)

    @pytest.mark.torch
    def test_multitask_regressor_integer_features(self):
        """Test MultitaskRegressor with integer features (fixes #3557)."""
        np.random.seed(123)
        X = np.random.randint(0, 2, size=(10, 50))
        y = np.random.rand(10, 2)
        dataset = dc.data.NumpyDataset(X, y)

        model = dc.models.MultitaskRegressor(
            n_tasks=2, n_features=50, layer_sizes=[32], dropouts=0.0)
        model.fit(dataset, nb_epoch=2)
        preds = model.predict(dataset)
        self.assertEqual(preds.shape, (10, 2, 1))

    @pytest.mark.torch
    def test_multitask_classifier_integer_features(self):
        """Test MultitaskClassifier with integer features (fixes #3557)."""
        np.random.seed(123)
        X = np.random.randint(0, 2, size=(10, 50))
        y = np.random.randint(0, 2, size=(10, 2))
        dataset = dc.data.NumpyDataset(X, y)

        model = dc.models.MultitaskClassifier(
            n_tasks=2,
            n_features=50,
            n_classes=2,
            layer_sizes=[32],
            dropouts=0.0)
        model.fit(dataset, nb_epoch=2)
        preds = model.predict(dataset)
        self.assertEqual(preds.shape, (10, 2, 2))

    @pytest.mark.torch
    def test_multitask_regressor_maccs_fingerprint(self):
        """Test MultitaskRegressor with MACCSKeysFingerprint directly."""
        smiles = [
            "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
            "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"
        ]
        featurizer = dc.feat.MACCSKeysFingerprint()
        features = featurizer.featurize(smiles)
        y = np.array([[3.112], [2.432]])
        dataset = dc.data.NumpyDataset(features, y)

        model = dc.models.MultitaskRegressor(
            n_tasks=1,
            n_features=features.shape[1],
            layer_sizes=[32],
            dropouts=0.0)
        model.fit(dataset, nb_epoch=2)
        preds = model.predict(dataset)
        self.assertEqual(preds.shape, (2, 1, 1))
