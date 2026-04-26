
def make_base_graph_params():
    return {
        # Number of local neighbors considered inside each batch; integer >= 1.
        "n_neighbors": 10,
        # Fraction of graph mass reserved for within-batch neighborhood structure; float from 0 to 1.
        "intra_fraction": 0.5,
        # Number of repeated Hungarian assignment rounds between each batch pair; integer >= 0.
        "n_inter_edges": 1,
        # Distance metric passed to sklearn.metrics.pairwise_distances; common options include
        # "euclidean", "cosine", "manhattan", and any metric accepted by sklearn/scipy.
        "metric": "euclidean",
        # Cross-batch assignment filter; options are None to keep all assignments, or a float in (0, 1].
        "assignment_quantile": 0.01,
        # Hubness correction before assignment and neighbor selection; options are "csls" or "none".
        "hubness_correction": "csls",
        # Number of local distances used for CSLS scale estimation; integer >= 1.
        "hubness_k": 10,
        # Convert corrected distances to reciprocal row/column rank distances before edge selection.
        "rank_correction": True,
        # Edge weights passed to UMAP/spectral embedding; options are "binary" or "distance".
        "edge_weighting": "binary",
        # Within-batch neighbor rule; True keeps only reciprocal neighbors, False keeps the union.
        "mutual_neighbors": False,
        # Within-batch neighbor scoring; options are "rank" or "distance".
        "neighbor_mode": "distance",
        # Return an undirected graph for downstream spectral/UMAP embedding; options are True or False.
        "symmetrize": True,
    }
    
def make_base_graph_params():
    return {
        # Number of local neighbors considered inside each batch; integer >= 1.
        "n_neighbors": 17,
        # Fraction of graph mass reserved for within-batch neighborhood structure; float from 0 to 1.
        "intra_fraction": 0.5,
        # Number of repeated Hungarian assignment rounds between each batch pair; integer >= 0.
        "n_inter_edges": 5,
        # Distance metric passed to sklearn.metrics.pairwise_distances; common options include
        # "euclidean", "cosine", "manhattan", and any metric accepted by sklearn/scipy.
        "metric": "euclidean",
        # Cross-batch assignment filter; options are None to keep all assignments, or a float in (0, 1].
        "assignment_quantile": 0.3,
        # Hubness correction before assignment and neighbor selection; options are "csls" or "none".
        "hubness_correction": "csls",
        # Number of local distances used for CSLS scale estimation; integer >= 1.
        "hubness_k": 10,
        # Convert corrected distances to reciprocal row/column rank distances before edge selection.
        "rank_correction": True,
        # Edge weights passed to UMAP/spectral embedding; options are "binary" or "distance".
        "edge_weighting": "binary",
        # Within-batch neighbor rule; True keeps only reciprocal neighbors, False keeps the union.
        "mutual_neighbors": False,
        # Within-batch neighbor scoring; options are "rank" or "distance".
        "neighbor_mode": "distance",
        # Return an undirected graph for downstream spectral/UMAP embedding; options are True or False.
        "symmetrize": True,
    }
