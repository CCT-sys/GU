# Geometric Unlearning

This directory focuses on **geometric unlearning** training and evaluation. It is built on PyTorch Lightningand performs “unlearning” on LLM. The core idea is to use **fuzzy anchor representations** (hidden states triggered by refusal responses) to localize tokens to forget, and apply corresponding unlearning/retaining constraints during training.

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

## Evaluation & quick check

The output model could use the OpenUnlearning to do all the evaluation

