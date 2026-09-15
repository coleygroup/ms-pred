"""Model versions, kept separate from the ms_pred package version."""

MODEL_REGISTRY = {
    "iceberg": {
        "version": "2.1",
        "reference": "doi:10.1101/2025.05.28.656653",
        "previous": {
            "2.0": {"branch": "iceberg_2.0", "reference": "doi:10.1101/2025.05.28.656653"},
            "1.0": {"branch": "iceberg_1.0", "reference": "doi:10.1021/acs.analchem.3c04654"},
        },
    },
    "scarf": {"version": "1.0", "reference": "arXiv:2303.06470"},
    "marason": {"version": "1.0", "reference": "arXiv:2502.17874"},
    "glacier": {"version": "1.0", "reference": "arXiv:2606.29161"},
}
