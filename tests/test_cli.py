from __future__ import annotations

from anndata import read_h5ad
import pytest

from clasp.cli import main


def test_cli_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "CLASP command line tools" in capsys.readouterr().out


def test_cli_embed_writes_h5ad(tmp_path, toy_adata):
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "embedded.h5ad"
    toy_adata.write_h5ad(input_path)

    status = main(
        [
            "embed",
            str(input_path),
            str(output_path),
            "--batch-key",
            "batch",
            "--label-key",
            "label",
            "--preset",
            "balanced",
            "--n-top-genes",
            "none",
            "--min-gene-counts",
            "0",
            "--normalize",
            "false",
            "--n-components",
            "4",
            "--n-neighbors",
            "4",
            "--n-inter-edges",
            "1",
            "--assignment-quantile",
            "1.0",
            "--embedding-method",
            "spectral",
        ]
    )

    result = read_h5ad(output_path)
    assert status == 0
    assert "X_clasp" in result.obsm
    assert result.obsm["X_clasp"].shape == (toy_adata.n_obs, 2)
    assert result.uns["clasp"]["graph"]["parameters"]["n_neighbors"] == 4


def test_cli_embed_accepts_custom_embedding_key_and_figure(tmp_path, toy_adata):
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "embedded.h5ad"
    figure_path = tmp_path / "embedding.pdf"
    toy_adata.write_h5ad(input_path)

    main(
        [
            "embed",
            str(input_path),
            str(output_path),
            "--batch-key",
            "batch",
            "--label-key",
            "label",
            "--embedding-key",
            "X_test_clasp",
            "--figure",
            str(figure_path),
            "--n-top-genes",
            "none",
            "--min-gene-counts",
            "0",
            "--normalize",
            "false",
            "--n-components",
            "4",
            "--n-neighbors",
            "4",
            "--n-inter-edges",
            "1",
            "--assignment-quantile",
            "1.0",
            "--embedding-method",
            "spectral",
        ]
    )

    result = read_h5ad(output_path)
    assert "X_test_clasp" in result.obsm
    assert figure_path.exists()
