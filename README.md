# Geometric Unlearning

This directory focuses on **geometric unlearning** training and evaluation. It is built on PyTorch Lightningand performs “unlearning” on LLM. The core idea is to distills a compact, low-rank safe-behavior subspace from a small set of safe reference prompts and uses lightweight anchor-in-context synthetic prompts to trigger localized, projection-based alignment of hidden representations to this safe subspace.

## Layout (geometric unlearning related)

- `run.py`: training entry; reads config and selects the `method_name = "fuzzy"` implementation.
- `method/hs_unlearn.py`: geometric unlearning LightningModule (`llama_fuzzy`).
- `configs/example.json`: typical training configs.
- `data/`: forget/retain/validation datasets and entity name lists.
- `test_fuzzy.py`: collects fuzzy anchor representations (outputs `.pt`).

## Environment & dependencies

Use a dedicated virtual environment. Core dependencies:

Install:

```bash
pip install -r requirements.txt
pip install peft deepspeed
```

## Data preparation

Training/validation data are specified in the config. Common fields:

- `train_set`: forget training set (JSON).
- `fuzzy_entity_names_path`: forget entity name list (JSON).
- `retain_names_path`: retain entity name list (JSON).
- `valid_sets` / `valid_type_path`: validation sets and their types.

Example files are under `data/` (e.g., `forget20-fictional.json`, `forget20-name.json`, `retain40_names.json`).

## Generate fuzzy anchor representations

Geometric unlearning expects `fuzzy_repr_path` to point to a `.pt` file. You can generate it with `test_fuzzy.py`:

1) Edit `MODEL_PATH` and `OUTPUT_PT` in `test_fuzzy.py`.
2) Run the script:

```bash
python test_fuzzy.py
```

Then set in your config:

```json
"fuzzy_repr_path": "/path/to/fuzzy_repr_input_anchor.pt"
```

## Train

Run with a config:

```bash
python run.py --config configs/fuzzy_unlearn.json
```


## Key fuzzy configs

The most relevant fields:

- `method_name`: must be `"fuzzy"` to use `llama_fuzzy`.
- `fuzzy_repr_path`: fuzzy anchor representation file.
- `fuzzy_entity_names_path` / `retain_names_path`: forget/retain entities.
- `forget_window`, `target_layers`, `w_dir`, `alpha_energy`, `w_retain_align`: forgetting direction and energy constraints.
- `forget_end_step`, `forget_hit_boost`, `forget_mask_decay`: forgetting schedule and hit weighting.
- `strategy`: typically `deepspeed_stage_2`.

## HF export (optional)

If you set:

```json
"hf_export_dir": "/path/to/export",
"hf_export_merge_lora": true,
"hf_export_dtype": "bf16"
```

The trainer will export HF weights to `hf_export_dir` after training (optionally merging LoRA).


## OpenUnlearning integration optional

This method can also be integrated into the OpenUnlearning framework [1] for unified training and evaluation.

To add this method to OpenUnlearning, the geometric unlearning logic should be implemented as a custom unlearning trainer, e.g., `GeometricUnlearnTrainer`, following OpenUnlearning's trainer interface. The trainer then needs to be registered in `TRAINER_REGISTRY` and exposed through a trainer config under `configs/trainer/`, where method-specific arguments such as `fuzzy_repr_path`, `target_layers`, `forget_window`, `w_dir`, `alpha_energy`, `w_retain_align`, `forget_end_step`, `forget_hit_boost`, and `forget_mask_decay` can be specified.

The data side also needs to be registered explicitly. Since this project uses synthetic forget/retain data and entity-name files, a corresponding synthetic data handler should be added or adapted in OpenUnlearning's `src/data/` module, registered in `DATASET_REGISTRY`, and referenced from `configs/data/datasets/`. The handler should load the forget set, retain set, fuzzy entity names, retain names, and validation metadata in the format expected by the geometric unlearning trainer.

A typical OpenUnlearning integration therefore requires:

1. Add a custom trainer, for example:
   - `src/trainer/unlearn/geometric_unlearn.py`
   - class name: `GeometricUnlearnTrainer`

2. Register the trainer:
   - import the trainer in the trainer registry module
   - call `_register_trainer(GeometricUnlearnTrainer)`

3. Add a trainer config:
   - `configs/trainer/GeometricUnlearn.yaml`
   - include both HuggingFace `TrainingArguments` and geometric-unlearning-specific `method_args`

4. Add or adapt a synthetic dataset handler:
   - load forget/retain JSON data

5. Register the synthetic dataset handler:
   - call `_register_data(SyntheticGeometricUnlearnDataset)`

6. Add dataset configs:
   - define forget, retain, and validation paths under `configs/data/datasets/`

7. Run training through OpenUnlearning's config system:
   - select the custom trainer config
   - select the synthetic geometric-unlearning dataset config
   - point `fuzzy_repr_path` to the generated fuzzy anchor representation `.pt` file

[1] Dorna V, Mekala A, Zhao W, et al. Openunlearning: Accelerating llm unlearning via unified benchmarking of methods and metrics[J]. Advances in Neural Information Processing Systems, 2026, 38.