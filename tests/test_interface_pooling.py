"""
Tests for interface pooling functionality in GATRanker.
"""
import torch
import pytest
from torch_geometric.data import Data, Batch
from src.models.gat_ranker import GATRanker


class TestInterfacePooling:
    """Test interface pooling modes."""

    def test_interface_pooling_all_mode(self):
        """Test interface pooling in 'all' mode."""
        model = GATRanker(
            node_feature_dim=10,
            hidden_dim=8,
            num_layers=1,
            interface_pool_mode="all"
        )

        # Create test data: 2 graphs
        # Graph 0: 4 nodes (2 interface, 2 non-interface)
        # Graph 1: 3 nodes (1 interface, 2 non-interface)

        # Graph 0
        x0 = torch.randn(4, 10)
        edge_index0 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        is_interface0 = torch.tensor([1, 1, 0, 0], dtype=torch.long)  # First 2 are interface
        batch0 = torch.zeros(4, dtype=torch.long)

        # Graph 1
        x1 = torch.randn(3, 10)
        edge_index1 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        is_interface1 = torch.tensor([1, 0, 0], dtype=torch.long)  # First 1 is interface
        batch1 = torch.ones(3, dtype=torch.long)

        # Combine into batch
        x = torch.cat([x0, x1], dim=0)
        edge_index = torch.cat([edge_index0, edge_index1 + 4], dim=1)  # Offset edge indices
        is_interface = torch.cat([is_interface0, is_interface1], dim=0)
        batch = torch.cat([batch0, batch1], dim=0)

        # Forward pass
        scores = model.forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            is_interface=is_interface
        )

        assert scores.shape == (2,), f"Expected (2,) scores, got {scores.shape}"

    def test_interface_pooling_split_roles_mode(self):
        """Test interface pooling in 'split_roles' mode."""
        model = GATRanker(
            node_feature_dim=10,
            hidden_dim=8,
            num_layers=1,
            interface_pool_mode="split_roles"
        )

        # Create test data: 1 graph with mixed antibody/antigen interface nodes
        x = torch.randn(6, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                                  [1, 2, 3, 4, 5, 0]], dtype=torch.long)

        # 2 antibody interface, 1 antibody non-interface, 2 antigen interface, 1 antigen non-interface
        is_interface = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.long)
        chain_role = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)  # 0=antibody, 1=antigen
        batch = torch.zeros(6, dtype=torch.long)

        # Forward pass
        scores = model.forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            is_interface=is_interface,
            chain_role=chain_role
        )

        assert scores.shape == (1,), f"Expected (1,) scores, got {scores.shape}"

    def test_interface_pooling_fallback_no_interface(self):
        """Test fallback when graph has no interface nodes."""
        model = GATRanker(
            node_feature_dim=10,
            hidden_dim=8,
            num_layers=1,
            interface_pool_mode="all"
        )

        # Create test data: 1 graph with no interface nodes
        x = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        is_interface = torch.zeros(4, dtype=torch.long)  # No interface nodes
        batch = torch.zeros(4, dtype=torch.long)

        # Forward pass should not crash and fall back to all-node pooling
        scores = model.forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            is_interface=is_interface
        )

        assert scores.shape == (1,), f"Expected (1,) scores, got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores should not be NaN"

    def test_interface_pooling_edge_types(self):
        """Test that edge types are handled correctly."""
        model = GATRanker(
            node_feature_dim=10,
            hidden_dim=8,
            num_layers=1,
            use_edge_types=True,
            num_edge_types=2
        )

        # Create test data with edge types
        x = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)  # Alternate covalent/noncovalent
        is_interface = torch.tensor([1, 1, 0, 0], dtype=torch.long)
        batch = torch.zeros(4, dtype=torch.long)

        # Forward pass
        scores = model.forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_type=edge_type,
            is_interface=is_interface
        )

        assert scores.shape == (1,), f"Expected (1,) scores, got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores should not be NaN"

    def test_forward_batch_compatibility(self):
        """Test forward_batch method works with new interface fields."""
        model = GATRanker(
            node_feature_dim=10,
            hidden_dim=8,
            num_layers=1,
            interface_pool_mode="all"
        )

        # Create test Batch
        data1 = Data(
            x=torch.randn(3, 10),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_type=torch.tensor([0, 1, 0], dtype=torch.long),
            is_interface=torch.tensor([1, 0, 1], dtype=torch.long),
            chain_role=torch.tensor([0, 1, 0], dtype=torch.long),
            y=torch.tensor([1.5])
        )

        data2 = Data(
            x=torch.randn(2, 10),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_type=torch.tensor([1, 0], dtype=torch.long),
            is_interface=torch.tensor([0, 1], dtype=torch.long),
            chain_role=torch.tensor([1, 0], dtype=torch.long),
            y=torch.tensor([2.0])
        )

        batch = Batch.from_data_list([data1, data2])

        # Forward pass
        scores = model.forward_batch(batch)

        assert scores.shape == (2,), f"Expected (2,) scores, got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores should not be NaN"


if __name__ == "__main__":
    # Run basic tests
    test = TestInterfacePooling()
    test.test_interface_pooling_all_mode()
    test.test_interface_pooling_split_roles_mode()
    test.test_interface_pooling_fallback_no_interface()
    test.test_interface_pooling_edge_types()
    test.test_forward_batch_compatibility()
    print("All interface pooling tests passed!")
