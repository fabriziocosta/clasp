"""Notebook runners for CLASP and IMSTE single-cell workflows."""

from __future__ import annotations


def run_clasp_imste_all_datasets(
    dataset_paths,
    *,
    single_cell,
    workflow,
    visualization,
    imste_embedder_class,
    clasp_params,
    preprocess_params,
    umap_params,
    imste_params,
    hvg_count,
    pc_count,
    max_cells,
    random_state=42,
    renderer="matplotlib",
    datashader_background="white",
    datashader_max_dim=900,
):
    """Fit and plot CLASP, UMAP, and IMSTE for each selected H5AD dataset."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if renderer not in {"matplotlib", "datashader"}:
        raise ValueError("renderer must be 'matplotlib' or 'datashader'.")

    current_preprocess = {**preprocess_params, "n_top_genes": hvg_count, "scale": True}
    current_clasp = {**clasp_params, "pca_components": pc_count}
    knn_params = {
        "k": 7,
        "cv_neighbors": 5,
        "cv_splits": 5,
        "random_state": random_state,
        "distance_std_threshold": 1.0,
        "alpha_range": (0.2, 0.8),
        "legend_position": "bottom",
    }
    cell_limit = "all" if max_cells is None else f"up to {max_cells:,}"
    hvg_label = "all" if hvg_count is None else f"{hvg_count:,}"
    print(f"Preprocessing: {hvg_label} HVGs, scaled, then {pc_count} PCs; {cell_limit} cells per dataset")

    def category_colors(values):
        categories = pd.Categorical(values).categories
        palette = plt.get_cmap("tab20", max(len(categories), 1))
        colors = {str(value): palette(i) for i, value in enumerate(categories)}
        if pd.isna(values).any():
            colors["Missing"] = (0.5, 0.5, 0.5, 1.0)
        return colors

    def plot_embedding(embedding, obs, key, colors, ax, title):
        if renderer == "datashader":
            from matplotlib.colors import to_hex

            color_key = {str(key): to_hex(value) for key, value in colors.items()}
            color_key["(missing)"] = color_key.get("Missing", "#808080")
            image = visualization.render_datashader_embedding(
                embedding,
                obs[key],
                label_key=key,
                color_key=color_key,
                width=datashader_max_dim,
                height=round(datashader_max_dim * 700 / 900),
                min_alpha=50,
                background=datashader_background,
                spread_px=1,
            )
            ax.imshow(image.to_pil())
            ax.set_title(title)
            ax.set_axis_off()
        else:
            single_cell.plot_embedding(
                embedding,
                obs,
                color_keys=(key,),
                point_size=7,
                point_alpha=0.7,
                ax=ax,
                legend_position="bottom",
                label_colors=colors,
            )
            ax.set_title(title)

    for number, dataset_path in enumerate(dataset_paths, start=1):
        print(f"\n[{number}/{len(dataset_paths)}] {dataset_path.name}")
        X, obs, _ = single_cell.load_dataset(
            path=dataset_path,
            matrix_key="X",
            obs_keys=tuple(single_cell.available_obs_columns(dataset_path)),
            max_cells=max_cells,
            random_state=random_state,
        )
        label_key = single_cell.default_label_key(obs)
        obs = obs.copy()
        batch_key = workflow.infer_batch_key(obs)
        label_colors = category_colors(obs[label_key])
        batch_colors = category_colors(obs[batch_key])

        print(
            f"Cells: {len(obs):,}; genes before HVG selection: {X.shape[1]:,}; "
            f"label: {label_key}; batch: {batch_key}"
        )
        processed, graph, umap_embedding, imste_embedding, _ = workflow.fit_clasp_umap_imste(
            X,
            obs,
            label_key=label_key,
            batch_key=batch_key,
            clasp_params=current_clasp,
            preprocess_params=current_preprocess,
            umap_params=umap_params,
            imste_params=imste_params,
            random_state=random_state,
            imste_embedder_class=imste_embedder_class,
        )
        actual_pcs = processed.obsm["X_pca"].shape[1]
        print(
            f"HVGs used: {processed.n_vars:,}; PCs used: {actual_pcs}; "
            f"CLASP graph: {graph.shape} ({graph.nnz:,} directed nonzero entries); "
            f"UMAP: {umap_embedding.shape}; IMSTE: {imste_embedding.shape}"
        )

        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        for row, (method, embedding) in enumerate(
            (("UMAP", umap_embedding), ("IMSTE", imste_embedding))
        ):
            plot_embedding(
                embedding,
                obs,
                batch_key,
                batch_colors,
                axes[row, 0],
                f"{dataset_path.stem} · {method} — batch provenance ({batch_key})",
            )
            plot_embedding(
                embedding,
                obs,
                label_key,
                label_colors,
                axes[row, 1],
                f"{dataset_path.stem} · {method} — {label_key}",
            )
            visualization.plot_knn_union(
                embedding, obs, label_key=label_key, ax=axes[row, 2], **knn_params
            )
            axes[row, 2].set_title(f"{dataset_path.stem} · {method} kNN union")

        fig.suptitle(f"Embedding renderer: {renderer}", y=1.01)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
