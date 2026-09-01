import torch

from transinterp.extraction.features import (
    SparseAutoencoder,
    fit_pca,
    top_activating_examples,
)


def test_fit_pca_reconstructs_low_rank_data_exactly():
    torch.manual_seed(0)
    basis_vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coefficients = torch.randn(50, 2)
    hidden = coefficients @ basis_vectors  # exactly rank-2 inside a 3D space

    basis = fit_pca(hidden, n_components=2)
    reconstruction = basis.reconstruct(basis.project(hidden))

    assert basis.components.shape == (2, 3)
    assert torch.allclose(reconstruction, hidden, atol=1e-3)
    assert basis.explained_variance_ratio.sum().item() <= 1.0 + 1e-5
    assert basis.explained_variance_ratio[0] >= basis.explained_variance_ratio[1]


def test_fit_pca_validates_component_count():
    hidden = torch.randn(10, 4)
    try:
        fit_pca(hidden, n_components=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_components > d_model")


def test_sparse_autoencoder_training_reduces_reconstruction_loss():
    torch.manual_seed(0)
    hidden = torch.randn(32, 4)
    sae = SparseAutoencoder(d_model=4, n_features=8)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.05)

    initial_loss = sae.loss(hidden, l1_coefficient=1e-4)["reconstruction_loss"].item()
    for _ in range(200):
        optimizer.zero_grad()
        losses = sae.loss(hidden, l1_coefficient=1e-4)
        losses["total_loss"].backward()
        optimizer.step()
    final_loss = sae.loss(hidden, l1_coefficient=1e-4)["reconstruction_loss"].item()

    assert final_loss < initial_loss


def test_sparse_autoencoder_rejects_undercomplete_dictionary():
    try:
        SparseAutoencoder(d_model=8, n_features=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_features < d_model")


def test_top_activating_examples_ranks_by_score():
    scores = torch.tensor([[0.1, 5.0], [0.9, 1.0], [0.2, 9.0]])
    tokens = ["a", "b", "c"]
    top = top_activating_examples(scores, tokens, feature_index=1, k=2)
    assert [token for token, _ in top] == ["c", "a"]
