# Input data

Raw TIFFs and the two NPZ caches are distributed as assets of the
`r1-inputs-20261109` release. From the `DL_XRD_R1` package root, run
`python data/download_inputs.py` to restore them at the paths below and verify
their checksums. These release assets are the data for this R1 package;
they are stored in GitHub Releases rather than duplicated as Git blobs.
Input manifests, acquisition metadata, checksums, and split
assignments are versioned with the code.

This folder contains model inputs, immutable sample metadata, and partition definitions.

The 1,901 TIFF inputs comprise 901 simulated, 800 known-class experimental, 100 noise-only and 100 SiC images. Simulated images are 1024 by 1024; experimental images are 1062 by 1028. Original acquisition instructions are included for the 20 experimental exposure folders. The available Al2O3 and unidentified OOD groups were not used in the requested comparison and are excluded. Duplicate copies of the noise/SiC images are not added.

`cache/xrd_dataset_cache.npz` stores image and label arrays. Its 0-D `batch_size` array is an input setting. `cache/fixed_images.npz` contains the same 1,701 known-class images in stable sample-ID order, plus labels, sample IDs, domains and materials. Use `allow_pickle=False` when reading either file.

`partitions/continual_partition.csv` defines the continual-learning scan roles, block assignments, and registered 10.0% experimental training-label mask. The loader verifies the partition hash before reading it. Every strategy, seed and order shares the same mask; registered run-profile CSVs contain it explicitly, and material-disjoint remapping retains the same label identities. Validation labels are separate from the training-label budget.

`manifests/fixed/scan.csv` and `grouped.csv` are NEW paper-matched fixed-model partitions shared by CNN, CNN-SNGP and CNN-SNGP-adapt. They use requested 10.0% experimental training anchors. Within each eligible domain/class, floor(10.0%) is assigned to validation, floor(10.0%) to internal testing and the rest to training. Training anchors are sampled only from target-training rows, with classwise round-half-up counts (at least one per nonempty class). Nontraining `training_label` fields are -1; `label` is reserved for evaluation truth. No geometry- or test-result-based partition selection is used.

The scan protocol excludes simulated Ag/Ti/V and experimental Cu as separate outer tests. The grouped protocol holds each of Al/Cu/Fe/Mo/Nb/Ni/Ti/Zn out in both domains, then applies the within-development split. Three repeats use initialization seeds 42/43/44, partition seeds 142/143/144 and anchor seeds 242/243/244. Every model receives the same manifest row identities. Exact counts are in `manifests/fixed/partition_counts.csv`.

`manifests/ood.csv` identifies all 200 confirmed noise/SiC scans for inference after model selection is locked. These inputs have no BCC/FCC/HCP reference label and are excluded from optimization, validation and GP precision updates.

`metadata/sample_inventory.csv` and `acquisition_metadata.csv` report the verified measurement hierarchy. Each known material has one owner-confirmed specimen, shared between its 1 s and 3 s exposures; this is eight known specimen groups, not 800 independent specimens. Scan positions come from acquisition instructions; coordinate units and independent session boundaries are unverified. SiC polytype/crystallinity are not inferred from its two exposure folders.

`cache/cache_manifest.json` records input hashes, shapes and partition policies; `continual/recipe_provenance.json` records the 10.0% experimental training-label setting. `metadata/raw_file_hashes.csv` provides SHA256 for every included TIFF. Files are independent copies, never references to the development workspace.

From the package root, run `python data/verify_inputs.py --full` to verify every raw TIFF and all source cache/partition identities. Run `python continual/verify_inputs.py` to verify the registered continual training-label mask and stage scopes. Run `python data/prepare_fixed_manifests.py --check` to regenerate the fixed partitions in memory and compare all rows. An explicit `--output-dir PATH` writes regenerated CSVs to a chosen directory; the default check does not change inputs.

The shared `data.preprocessing.load_image` function implements the original preprocessing. `python -m data.preprocessing --rebuild-fixed-cache NEW_PATH.npz` rebuilds a new fixed-image input cache directly from the included TIFFs and verifies every image against the immutable float32 fingerprints before writing. It never overwrites an existing output. Bitwise fingerprint checks passed for 52 representative TIFFs across all material identities and both domains. A few raw simulated TIFFs contain positive infinite peaks; the original percentile upper clipping removes these before resizing.
