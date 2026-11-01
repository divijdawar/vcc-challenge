"""All settings are attributes of config. Paths resolve relative to this file.

Importing this file does not load datasets, initialize weights, or start training.
"""


class config:
    # Reproducibility and data preparation.
    seed = 42
    project_root = ".."
    output_dir = "artifacts/observations/vcc2025_vcc2026_v1"
    chunk_rows = 256
    gene_identifier_namespace = "gene_symbol"
    control_reference_fraction = 0.2

    # Contexts, labels, targets, and gene order come from the release files.
    release_enabled = True
    release_data_dir = "../data"
    release_manifest = "manifest.json"
    release_gene_file = "gene_names.csv"
    release_gene_column = "gene_name"
    release_target_file = "pert_counts.csv"
    release_target_column = "target_gene"
    release_file_template = "context_{context}.h5ad"
    unperturbed_dataset = "vcc2026"

    # Only the official 2025 training release supplies supervised responses.
    # Download separately; this program never downloads datasets.
    perturbed_dataset = "vcc2025"
    response_source_id = "vcc2025_H1"
    sources = [{
        "id": "vcc2025_H1", "kind": "supervised", "dataset": "vcc2025",
        "release_split": "train", "context": "H1",
        "path": "inputs/vcc2025/adata_Training.h5ad",
        "counts_are_raw": True, "counts_location": "X",
        "gene_identifier_namespace": "gene_symbol", "gene_key": "_index",
        "target_key": "target_gene", "control_labels": ["non-targeting"],
        "batch_key": "batch",
    }]
    # A/B/C are anonymous cell lines, not verified matches to H1. This is an
    # explicit experimental transfer policy; never relabel them as H1.
    control_context_policy = "uniform_transfer"  # or "verified_mapping"
    verified_context_map = {}  # used only with verified_mapping
    # None uses the release axis. Supply explicit lists after freezing splits.
    encoder_genes = None
    output_genes = None

    # Dimensions must match frozen gene axes and the training target vocabulary.
    input_dim = 18533
    output_dim = 18533
    condition_dim = 300
    hidden_dim = 256
    invariant_dim = 85
    responsive_dim = 20
    n_layers = 3
    negative_slope = 0.2
    append_measurement_mask = True
    sample_logvar_min = -20.0
    sample_logvar_max = 20.0
    consistent_logvar_clamp = False
    zero_initialize_structure = False

    # Independent controls use MMD; calibrate its bandwidth before training.
    # Choose reference_pairwise only for explicit reference reproduction.
    alignment = "mmd"
    mmd_bandwidth = None
    reconstruction_weight = 1.0
    responsive_kl_weight = 0.01
    invariant_kl_weight = 0.01
    alignment_weight = 0.05
    sparsity_weight = 1e-6

    # Residual count training. Select "reference" for the original objective.
    training_objective = "residual"
    population_weight = 1.0
    normalization_target_sum = 10000.0
    population_target_sum = 50000.0

    # Local prediction/export; set prediction_bundle after fitting a model.
    prediction_bundle = None
    prediction_output = "artifacts/predictions/prediction.h5ad"
    prediction_device = None
    export_batch_rows = 64
    cells_per_target = 400
    max_counts_per_cell = 1000000
    max_submission_cells = 400000
    max_submission_entries = 4750000000

    # Optimization. Set planned_steps from the frozen training population.
    learning_rate = 1e-4
    gradient_clip = 0.5
    warmup_fraction = 0.25
    warmup_curve = "linear"
    planned_steps = None
    allow_unpaired_reference_alignment = False
    diagnostics_directory = None
    # Capture gradients and Adam separately so nonfinite steps stop before updates.
    training_jit = True
    jit_cache_size = 4
    observation_cache_bytes = 256 * 1024 * 1024  # per reader, raw + normalized CSR
    prefetch_steps = 2  # CPU observation steps; tensors stay on the training thread

    # Reproducible target/cell splits within the 2025 training release only.
    split_manifest = "artifacts/splits/vcc2025_v1.json"
    test_target_fraction = 0.2
    development_target_fraction = 0.1
    development_cell_fraction = 0.1
    min_response_cells = 64
    max_response_cells_per_target = 256

    # Runnable training entry point. Dimensions/vocabulary come from frozen axes.
    training_device = None  # DEV=CUDA selects the tinygrad CUDA backend
    training_epochs = 20  # used when planned_steps is None
    groups_per_step = 4
    run_directory = "artifacts/runs/vcc2025_to_vcc2026"
    log_interval = 10
    validation_interval = 100
    checkpoint_interval = 500
    validation_groups = 8
    validation_seed = 4242
    population_max_controls = 8192
    verify_cache_before_training = True

    # Sampling and condition encoding.
    cells_per_group = 64
    max_batch_rows = 256
    control_label = "non-targeting"
    include_control_indicator = False

