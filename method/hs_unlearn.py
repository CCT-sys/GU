import os, json, copy, re, string
import math
from typing import List
from typing import Optional
from difflib import SequenceMatcher
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, Dataset

import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"


class llama_fuzzy(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)


        self.cache_dir = hparams.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, use_fast=True, cache_dir=self.cache_dir
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; check tokenizer_name_or_path.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = None
        config_path = os.path.join(hparams.model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            if "quantization_config" in config_dict:
                del config_dict["quantization_config"]
            model_type = config_dict.get("model_type", None)
            if model_type:
                config_dict = dict(config_dict)
                config_dict.pop("model_type", None)
                model_config = AutoConfig.for_model(model_type, **config_dict)

        base_model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,        
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            config=model_config
        )


        base_model.resize_token_embeddings(len(self.tokenizer))
        base_model.config.eos_token_id = self.tokenizer.eos_token_id
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(base_model, "generation_config", None) is not None:
            base_model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.teacher = copy.deepcopy(base_model).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False


        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=getattr(hparams, "lora_r", 8),
            lora_alpha=getattr(hparams, "lora_alpha", 16),
            lora_dropout=getattr(hparams, "lora_dropout", 0.1),
            target_modules=getattr(hparams, "lora_target_modules", ["v_proj", "o_proj"])
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.model.print_trainable_parameters()
        self._printed_val_prompt = False


        self.mode = hparams.mode
        self.learning_rate = self.hparams.learning_rate

        self.target_layers = self._resolve_target_layers(getattr(hparams, "target_layers", None))


        self.basis_type      = getattr(hparams, "basis_type", "safe")  
        self.fuzzy_rank      = getattr(hparams, "fuzzy_rank", 4)
        self.use_householder = getattr(hparams, "use_householder", True)
        self.w_dir           = getattr(hparams, "w_dir", 0.6)
        self.alpha_energy    = getattr(hparams, "alpha_energy", 0.3)
        self.forget_window   = getattr(hparams, "forget_window", 8)
        self.retain_window   = getattr(hparams, "retain_window", 6)

        self.hit_include_token = bool(getattr(hparams, "hit_include_token", True))
        self.hit_both_sides = bool(getattr(hparams, "hit_both_sides", True))


        self.w_proto_safe = float(getattr(hparams, "w_proto_safe", 0.2))
        self.w_info_energy = float(getattr(hparams, "w_info_energy", 0.2))
        self.proto_k = int(getattr(hparams, "proto_k", 16))


        self.lambda_bg_keep = getattr(hparams, "lambda_bg_keep", 0.05)


        self.w_retain_align = getattr(hparams, "w_retain_align", 0.4)
        self.w_retain_hs = float(getattr(hparams, "w_retain_hs", 0.0))
        self.retain_kl_temperature = float(getattr(hparams, "retain_kl_temperature", 2.0))
        self.retain_only_steps = int(getattr(hparams, "retain_only_steps", 0))

        # Forget schedule
        self.forget_warmup_steps = int(getattr(hparams, "forget_warmup_steps", 0))
        self.forget_ramp_steps = int(getattr(hparams, "forget_ramp_steps", 200))
        self.forget_ramp_min = float(getattr(hparams, "forget_ramp_min", 0.0))

        self.forget_end_step = int(getattr(hparams, "forget_end_step", -1))

        self.forget_hit_boost = bool(getattr(hparams, "forget_hit_boost", True))
        self.forget_hit_boost_max = float(getattr(hparams, "forget_hit_boost_max", 4.0))

        self.forget_mask_decay = float(getattr(hparams, "forget_mask_decay", 0.0))


        self.bg_keep_on_all_forget = bool(getattr(hparams, "bg_keep_on_all_forget", True))
        self.bg_keep_temperature = float(getattr(hparams, "bg_keep_temperature", self.retain_kl_temperature))


        self.w_forget_entropy = float(getattr(hparams, "w_forget_entropy", 0.0))
        self.forget_entropy_temperature = float(getattr(hparams, "forget_entropy_temperature", 1.0))

        self.w_lora_reg = float(getattr(hparams, "w_lora_reg", 0.0))


        self.max_name_span       = getattr(hparams, "max_name_span", 8)
        self.max_matches_per_seq = getattr(hparams, "max_matches_per_seq", 4)
        self.allow_fuzzy_match   = getattr(hparams, "allow_fuzzy_match", True)
        self.allow_fuzzy_match_forget = bool(getattr(hparams, "allow_fuzzy_match_forget", False))
        self.allow_fuzzy_match_retain = bool(getattr(hparams, "allow_fuzzy_match_retain", self.allow_fuzzy_match))
        self.max_name_span_forget = int(getattr(hparams, "max_name_span_forget", self.max_name_span))
        self.max_name_span_retain = int(getattr(hparams, "max_name_span_retain", self.max_name_span))
        self.debug_hits = bool(getattr(hparams, "debug_hits", False))

        self.refusal_text = getattr(hparams, "refusal_text", "I don't know.")
        self.w_refusal = float(getattr(hparams, "w_refusal", 0.1))

        self.w_cf_inv = float(getattr(hparams, "w_cf_inv", 0.0))
        self.cf_inv_use_tail = bool(getattr(hparams, "cf_inv_use_tail", True))

        self.cf_inv_include_hit = bool(getattr(hparams, "cf_inv_include_hit", False))
        _cf_layers = getattr(hparams, "cf_inv_layers", None)
        self.cf_inv_layers = self.target_layers if _cf_layers is None else self._resolve_target_layers(_cf_layers)

        self.forget_tail_to_eos = bool(getattr(hparams, "forget_tail_to_eos", False))


        _mask_id = getattr(hparams, "cf_mask_token_id", None)
        if _mask_id is None:
            _mask_id = self.tokenizer.unk_token_id if (self.tokenizer.unk_token_id is not None) else self.tokenizer.eos_token_id
        self.cf_mask_token_id = int(_mask_id)

        refusal_ids = self.tokenizer.encode(self.refusal_text, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            if (len(refusal_ids) == 0) or (refusal_ids[-1] != self.tokenizer.eos_token_id):
                refusal_ids = refusal_ids + [self.tokenizer.eos_token_id]
        self.refusal_token_ids = torch.tensor(refusal_ids, dtype=torch.long)


        self.auto_select_layers = bool(getattr(hparams, "auto_select_layers", False))
        self.auto_layer_topk = int(getattr(hparams, "auto_layer_topk", 2))
        self.auto_prefer_upper = bool(getattr(hparams, "auto_prefer_upper_layers", True))
        self.auto_min_hs = int(getattr(hparams, "auto_min_hs", 1))


        fuzzy_path = getattr(hparams, 'fuzzy_repr_path', 'fuzzy_repr.pt')
        self.fuzzy_repr_list = torch.load(fuzzy_path, map_location="cpu")
        self.fuzzy_repr_list = [t.to(torch.float32).cpu() for t in self.fuzzy_repr_list]
        if len(self.fuzzy_repr_list) == 0 or self.fuzzy_repr_list[0].ndim != 2:
            raise ValueError("fuzzy_repr_list blank")

        self.use_info_bank = bool(getattr(hparams, "use_info_bank", False))
        info_path = getattr(hparams, "info_repr_path", None)

        self.info_repr_list = None
        if self.use_info_bank:
            if (info_path is None) or (not os.path.exists(info_path)):
                raise ValueError("use_info_bank=True but info_repr_path is missing or not found.")
            self.info_repr_list = torch.load(info_path, map_location="cpu")
            self.info_repr_list = [t.to(torch.float32).cpu() for t in self.info_repr_list]
            if len(self.info_repr_list) == 0 or self.info_repr_list[0].ndim != 2:
                raise ValueError("info_repr_list is empty or has invalid shape. Expected [L,H] or [L+1,H].")


        tgt_path = getattr(hparams, 'fuzzy_entity_names_path', 'fuzzy_names.json')
        if os.path.exists(tgt_path):
            with open(tgt_path, 'r') as f:
                self.target_names = json.load(f)
            self.target_name_token_ids = []
            for n in self.target_names:
                self.target_name_token_ids.extend(self._encode_name_variants(n))
        else:
            self.target_names, self.target_name_token_ids = [], []


        self.retain_name_token_ids = []
        r_path = getattr(hparams, 'retain_names_path', None)
        if r_path and os.path.exists(r_path):
            with open(r_path, 'r') as f:
                retain_names = json.load(f)
            self.retain_name_token_ids = []
            for n in retain_names:
                self.retain_name_token_ids.extend(self._encode_name_variants(n))


        num_layers = self._infer_num_layers_from_model()
        rows = self.fuzzy_repr_list[0].shape[0]
        if rows == num_layers:
            self._repr_mode, self._repr_shift = "block", -1
        elif rows == num_layers + 1:
            self._repr_mode, self._repr_shift = "hidden_states", 0
        else:
            raise ValueError(f"fuzzy_repr 行数={rows} 不匹配 num_layers={num_layers}")

        if self.auto_select_layers:
            new_layers = self._select_layers_by_separability(num_layers)
            print(f"[auto_layers] picked(hs_idx)={new_layers}")
            self.target_layers = new_layers
        print(f"[align_layers] hs_idx={self.target_layers}")

        self._build_fuzzy_basis_cpu()

        self.early_stop_target_acc = float(getattr(hparams, "early_stop_target_acc", 0.12))
        self.early_stop_test_acc = float(getattr(hparams, "early_stop_test_acc", 0.32))
        self.early_stop_patience = int(getattr(hparams, "early_stop_patience", 2))
        self.early_stop_target_type = getattr(hparams, "early_stop_target_type", "target")
        self.early_stop_test_type = getattr(hparams, "early_stop_test_type", "nei")
        self._early_stop_ok_epochs = 0
        self._val_type_by_idx = []


    def _infer_num_layers_from_model(self):
        cfg = self.model.config
        for key in ["num_hidden_layers", "n_layer", "num_layers"]:
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return v
        for path in ["model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"]:
            obj = self.model
            ok = True
            for name in path.split("."):
                if hasattr(obj, name): obj = getattr(obj, name)
                else: ok=False; break
            if ok and hasattr(obj, "__len__"):
                return len(obj)
        raise ValueError("can't find the layer amount")

    def _resolve_target_layers(self, user_layers):
        num_layers = self._infer_num_layers_from_model()
        if user_layers is None:
            tl_hs = [num_layers - 2, num_layers - 1, num_layers]
        else:
            tl_hs = []
            for li in list(user_layers):
                li_block = li if li >= 0 else (num_layers + li)
                if not (0 <= li_block < num_layers):
                    raise ValueError(f"target_layers 越界: {li}")
                tl_hs.append(li_block + 1) 
            tl_hs = sorted(set(tl_hs))
        assert all(1 <= x <= num_layers for x in tl_hs)
        print(f"[layers] num_layers={num_layers}, target(hs_idx)={tl_hs}")
        return tl_hs

    def _encode_name(self, n: str):
        try:
            return self.tokenizer.encode(n, add_special_tokens=False, add_prefix_space=True)
        except TypeError:
            return self.tokenizer.encode(n, add_special_tokens=False)

    def _encode_name_variants(self, name: str) -> List[List[int]]:

        variants = [name, " " + name, "\n" + name, f"({name}", f"\"{name}", f"'{name}"]
        out: List[List[int]] = []

        for v in variants:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if ids and ids not in out:
                out.append(ids)

        try:
            ids2 = self.tokenizer.encode(name, add_special_tokens=False, add_prefix_space=True)
            if ids2 and ids2 not in out:
                out.append(ids2)
        except TypeError:
            pass

        return out

    @torch.no_grad()
    def _build_basis_and_proto_cpu(self, repr_list, hs_idx: int, prefix: str):

        idx_in_repr = hs_idx + self._repr_shift
        feats = [rep[idx_in_repr].to(torch.float32).cpu() for rep in repr_list]  # list of [H]
        X = torch.stack(feats, dim=0)                                            # [N,H]
        mu = X.mean(dim=0, keepdim=True)                                         # [1,H]
        Xc = X - mu

        k = min(self.fuzzy_rank, Xc.shape[0], Xc.shape[1])
        if k <= 0:
            raise ValueError("Invalid fuzzy_rank (must be >= 1).")

        try:
            _, _, V = torch.pca_lowrank(Xc, q=k, center=False)
            Vb = V[:, :k]
        except Exception:
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            Vb = Vh.t()[:, :k]

        d = ((X - mu).pow(2).mean(dim=1))  # [N]
        kk = min(int(self.proto_k), X.shape[0])
        idx = torch.topk(d, k=kk, largest=False).indices
        proto = X[idx].mean(dim=0)

        self.register_buffer(f"{prefix}_mu_L{hs_idx}", mu.squeeze(0), persistent=False)
        self.register_buffer(f"{prefix}_basis_L{hs_idx}", Vb, persistent=False)
        self.register_buffer(f"{prefix}_proto_L{hs_idx}", proto, persistent=False)

    @torch.no_grad()
    def _build_fuzzy_basis_cpu(self):
        for hs_idx in self.target_layers:
            self._build_basis_and_proto_cpu(self.fuzzy_repr_list, hs_idx, prefix="safe")

        if self.use_info_bank and (self.info_repr_list is not None):
            for hs_idx in self.target_layers:
                self._build_basis_and_proto_cpu(self.info_repr_list, hs_idx, prefix="info")

    @torch.no_grad()
    def _select_layers_by_separability(self, num_layers: int) -> List[int]:

        if (not self.use_info_bank) or (self.info_repr_list is None):
            return self.target_layers

        scores = []
        eps = 1e-6

        for hs_idx in range(1, num_layers + 1):
            if hs_idx < self.auto_min_hs:
                continue
            idx = hs_idx + self._repr_shift

            Xs = torch.stack([r[idx].float() for r in self.fuzzy_repr_list], dim=0)  # [Ns,H]
            Xi = torch.stack([r[idx].float() for r in self.info_repr_list], dim=0)   # [Ni,H]

            mu_s = Xs.mean(dim=0)
            mu_i = Xi.mean(dim=0)

            var_s = Xs.var(dim=0, unbiased=False).mean()
            var_i = Xi.var(dim=0, unbiased=False).mean()

            num = (mu_i - mu_s).pow(2).mean()
            den = var_i + var_s + eps
            sc = (num / den).item()
            scores.append((hs_idx, sc))

        if not scores:
            return self.target_layers

        if self.auto_prefer_upper:
            scores.sort(key=lambda x: (x[1], x[0]))
            scores = list(reversed(scores))
        else:
            scores.sort(key=lambda x: x[1], reverse=True)

        picked = sorted({hs for hs, _ in scores[: self.auto_layer_topk]})
        return picked

    def _forget_ramp(self) -> float:
        s = int(self.global_step)
        if s < self.forget_warmup_steps:
            return 0.0
        if self.forget_ramp_steps <= 0:
            return 1.0
        x = (s - self.forget_warmup_steps) / float(self.forget_ramp_steps)
        ramp = float(max(0.0, min(1.0, x)))
        return float(max(ramp, self.forget_ramp_min))

    def _safe_fp32(self, *tensors):
        outs = []
        for t in tensors:
            outs.append(t.float() if t is not None and t.dtype != torch.float32 else t)
        return outs if len(outs) > 1 else outs[0]

    def _nanfix(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    def _masked_kl(self, student_logits, teacher_logits, attn_mask, temperature=2.0):
        s_logp = F.log_softmax(student_logits / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits / temperature, dim=-1)
        kl_tok = F.kl_div(s_logp, t_prob, reduction="none").sum(-1) * (temperature * temperature)
        denom = attn_mask.float().sum().clamp_min(1.0)
        return (kl_tok * attn_mask.float()).sum() / denom

    def _masked_entropy_from_logits(self, logits, attn_mask, temperature: float = 1.0):

        x = logits / max(1e-6, float(temperature))
        logp = F.log_softmax(x, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(dim=-1)  # [B,T]
        denom = attn_mask.float().sum().clamp_min(1.0)
        return (ent * attn_mask.float()).sum() / denom

    def _lora_l2(self):
        reg = torch.zeros((), device=next(self.model.parameters()).device, dtype=torch.float32)
        for n, p in self.model.named_parameters():
            if p.requires_grad and ("lora" in n.lower()):
                reg = reg + p.float().pow(2).sum()
        return reg

    def _find_hits(self, src_ids, token_id_lists, L, device):
        hits = [[] for _ in range(src_ids.size(0))]
        if not token_id_lists: return hits

        def _rough_match(row, pat, j):
            if not self.allow_fuzzy_match:
                return torch.equal(row[j:j+len(pat)], pat)
            same = (row[j:j+len(pat)] == pat).sum().item()
            return same >= max(len(pat) - 1, int(0.7 * len(pat)))

        for name_token_ids in token_id_lists:
            if not name_token_ids: continue
            name_len = min(len(name_token_ids), self.max_name_span)
            pat = torch.tensor(name_token_ids[:name_len], device=device)
            for i in range(src_ids.size(0)):
                row = src_ids[i]
                count = 0
                for j in range(0, L - name_len + 1):
                    if _rough_match(row, pat, j):
                        hits[i].append(j + name_len - 1)
                        count += 1
                        if count >= self.max_matches_per_seq: break
        for i in range(src_ids.size(0)):
            hits[i] = sorted(set(hits[i]))
        return hits

    def _find_hits_cpu(self, src_ids_cpu: torch.Tensor, token_id_lists, L: int, allow_fuzzy_match=None, max_name_span=None):

        rows = src_ids_cpu.tolist()
        hits = [[] for _ in range(len(rows))]
        if not token_id_lists:
            return hits
        if allow_fuzzy_match is None:
            allow_fuzzy_match = self.allow_fuzzy_match
        if max_name_span is None:
            max_name_span = self.max_name_span

        for name_token_ids in token_id_lists:
            if not name_token_ids:
                continue
            name_len = min(len(name_token_ids), max_name_span)
            if name_len <= 0 or name_len > L:
                continue

            pat = name_token_ids[:name_len]
            if allow_fuzzy_match:
                thr = max(name_len - 1, int(0.7 * name_len))
            else:
                thr = name_len

            for i, row in enumerate(rows):
                cnt = 0
                for j in range(0, L - name_len + 1):
                    seg = row[j:j + name_len]
                    if allow_fuzzy_match:
                        same = sum(1 for a, b in zip(seg, pat) if a == b)
                        ok = (same >= thr)
                    else:
                        ok = (seg == pat)

                    if ok:
                        hits[i].append(j + name_len - 1)
                        cnt += 1
                        if cnt >= self.max_matches_per_seq:
                            break

        return [sorted(set(h)) for h in hits]

    def _find_spans_cpu(self, src_ids_cpu: torch.Tensor, token_id_lists, L: int, allow_fuzzy_match=None, max_name_span=None):

        rows = src_ids_cpu.tolist()
        spans = [[] for _ in range(len(rows))]
        if not token_id_lists:
            return spans
        if allow_fuzzy_match is None:
            allow_fuzzy_match = self.allow_fuzzy_match
        if max_name_span is None:
            max_name_span = self.max_name_span

        for name_token_ids in token_id_lists:
            if not name_token_ids:
                continue
            name_len = min(len(name_token_ids), max_name_span)
            if name_len <= 0 or name_len > L:
                continue
            pat = name_token_ids[:name_len]
            thr = max(name_len - 1, int(0.7 * name_len)) if allow_fuzzy_match else name_len

            for i, row in enumerate(rows):
                cnt = 0
                for j in range(0, L - name_len + 1):
                    seg = row[j:j + name_len]
                    if allow_fuzzy_match:
                        same = sum(1 for a, b in zip(seg, pat) if a == b)
                        ok = (same >= thr)
                    else:
                        ok = (seg == pat)
                    if ok:
                        spans[i].append((j, j + name_len - 1))
                        cnt += 1
                        if cnt >= self.max_matches_per_seq:
                            break

        out = []
        for sp in spans:
            sp2 = sorted(set(sp))
            out.append(sp2)
        return out

    def _mask_spans_cpu(self, ids_cpu: torch.Tensor, spans, mask_token_id: int, attn_mask_cpu: Optional[torch.Tensor] = None):

        B, L = ids_cpu.size()
        for i in range(B):
            for (s, e) in spans[i]:
                s = max(0, int(s))
                e = min(L - 1, int(e))
                if e < s:
                    continue
                if attn_mask_cpu is not None:
                    for j in range(s, e + 1):
                        if int(attn_mask_cpu[i, j].item()) == 1:
                            ids_cpu[i, j] = int(mask_token_id)
                else:
                    ids_cpu[i, s:e + 1] = int(mask_token_id)
        return ids_cpu

    def _tail_mask_from_hits_cpu(self, hits, attn_mask_cpu: torch.Tensor, include_hit: bool = False):

        B, L = attn_mask_cpu.size()
        m = torch.zeros_like(attn_mask_cpu, dtype=torch.float)
        for i in range(B):
            if not hits[i]:
                continue
            start = min(hits[i])
            if not include_hit:
                start = start + 1
            if start < 0:
                start = 0
            if start >= L:
                continue
            m[i, start:] = attn_mask_cpu[i, start:].float()
        return m

    def _window_mask_from_hits(self, hits, W, attn_mask, include_hit=True, both_sides=False, decay: float = 0.0):

        if W <= 0:
            return torch.zeros_like(attn_mask, dtype=torch.float)
        B, L = attn_mask.size()
        m = torch.zeros_like(attn_mask, dtype=torch.float)

        def w(dist: int) -> float:
            if dist == 0:
                return 1.0
            if decay is None or decay <= 0.0:
                return 1.0
            return float(math.exp(-float(dist) / float(decay)))

        for i in range(B):
            for p in hits[i]:
                # hit token
                if include_hit and 0 <= p < L and attn_mask[i, p] == 1:
                    m[i, p] = max(float(m[i, p].item()), w(0))

                # forward window
                for q in range(1, W + 1):
                    j = p + q
                    if j < L and attn_mask[i, j] == 1:
                        m[i, j] = max(float(m[i, j].item()), w(q))

                # backward window
                if both_sides:
                    for q in range(1, W + 1):
                        j = p - q
                        if j >= 0 and attn_mask[i, j] == 1:
                            m[i, j] = max(float(m[i, j].item()), w(q))
        return m


    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(self.hparams, "weight_decay", 0.01),
            betas=(0.9, 0.98)
        )
        return [opt]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=1.0, gradient_clip_algorithm="norm"):
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)


    def training_step(self, batch, batch_idx):
        device = next(self.model.parameters()).device

        f_ids_cpu = batch["forget_input_ids"]
        f_mask_cpu = batch["forget_attention_mask"]
        Bf, Lf = f_ids_cpu.size()

        tgt_hits = self._find_hits_cpu(
            f_ids_cpu,
            self.target_name_token_ids,
            Lf,
            allow_fuzzy_match=self.allow_fuzzy_match_forget,
            max_name_span=self.max_name_span_forget,
        )

        f_tok_mask_cpu = self._window_mask_from_hits(
            tgt_hits,
            int(max(0, self.forget_window)),
            f_mask_cpu,
            include_hit=self.hit_include_token,
            both_sides=self.hit_both_sides,
            decay=float(self.forget_mask_decay),
        ).float()
        if self.forget_tail_to_eos:
            tail_mask_cpu = self._tail_mask_from_hits_cpu(
                tgt_hits, f_mask_cpu, include_hit=self.hit_include_token
            )
            f_tok_mask_cpu = torch.maximum(f_tok_mask_cpu, tail_mask_cpu)

        hit_mask_b_cpu = (f_tok_mask_cpu.sum(dim=1) > 0)
        hit_idx_cpu = hit_mask_b_cpu.nonzero(as_tuple=False).squeeze(-1)
        has_forget = (hit_idx_cpu.numel() > 0)

        # move to GPU
        f_ids = f_ids_cpu.to(device, non_blocking=True)
        f_mask = f_mask_cpu.to(device, non_blocking=True)
        f_tok_mask = f_tok_mask_cpu.to(device, non_blocking=True)

        self.log("dbg/num_target_hits", torch.tensor(sum(len(h) for h in tgt_hits), device=device), prog_bar=True)
        self.log("dbg/forget_tokmask_mean", f_tok_mask.mean(), prog_bar=False)
        self.log("dbg/forget_tokmask_sum", f_tok_mask.sum(), prog_bar=True)
        self.log("dbg/forget_nonempty_seqs", (f_tok_mask.sum(dim=1) > 0).float().sum(), prog_bar=True)
        if self.debug_hits and batch_idx == 0 and int(self.global_step) == 0:
            tok_name = getattr(self.tokenizer, "name_or_path", "unknown")
            print(f"[dbg_hits] tokenizer={tok_name} padding_side={self.tokenizer.padding_side} input_length={Lf}")
            f_text = batch.get("forget_text", None)
            if f_text is not None:
                sample = f_text[0] if isinstance(f_text, (list, tuple)) else f_text
                sample = str(sample)
                matched = [n for n in self.target_names if n in sample][:5]
                print(f"[dbg_hits] forget_text_sample={sample[:200]}")
                print(f"[dbg_hits] matched_names={matched}")
                print(f"[dbg_hits] hits_in_sample={len(tgt_hits[0])} attn_len={int(f_mask_cpu[0].sum().item())}")

        # Only request hidden states if we will compute forget repr losses
        need_forget_hs = bool(has_forget and (len(self.target_layers) > 0))
        out_f = self.model(
            input_ids=f_ids,
            attention_mask=f_mask,
            output_hidden_states=need_forget_hs,
            return_dict=True
        )
        Hf = out_f.hidden_states if need_forget_hs else None

        hit_idx = hit_idx_cpu.to(device, non_blocking=True)
        forget_dir_list, forget_eng_list = [], []
        proto_loss_list, info_loss_list = [], []
        refusal_loss = torch.zeros((), device=device, dtype=torch.float32)
        entropy_loss = torch.zeros((), device=device, dtype=torch.float32)
        cf_inv_loss = torch.zeros((), device=device, dtype=torch.float32)

        if (self.w_cf_inv > 0) and has_forget and need_forget_hs:
            cf_spans = self._find_spans_cpu(
                f_ids_cpu,
                self.target_name_token_ids,
                Lf,
                allow_fuzzy_match=self.allow_fuzzy_match_forget,
                max_name_span=self.max_name_span_forget,
            )
            f_ids_cf_cpu = f_ids_cpu.clone()
            f_ids_cf_cpu = self._mask_spans_cpu(
                f_ids_cf_cpu, cf_spans, mask_token_id=self.cf_mask_token_id, attn_mask_cpu=f_mask_cpu
            )
            f_ids_cf = f_ids_cf_cpu.to(device, non_blocking=True)

            with torch.no_grad():
                out_cf = self.model(
                    input_ids=f_ids_cf,
                    attention_mask=f_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                Hcf = out_cf.hidden_states

            if self.cf_inv_use_tail:
                cf_mask_cpu = self._tail_mask_from_hits_cpu(
                    tgt_hits, f_mask_cpu, include_hit=self.cf_inv_include_hit
                )
            else:
                cf_mask_cpu = (f_tok_mask_cpu > 0).float()

            cf_mask = cf_mask_cpu.to(device, non_blocking=True)[hit_idx]
            denom = cf_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mask3d = cf_mask.unsqueeze(-1)

            per_layer = []
            for hs_idx in self.cf_inv_layers:
                h_o = self._safe_fp32(Hf[hs_idx][hit_idx])
                h_c = self._safe_fp32(Hcf[hs_idx][hit_idx])
                v_o = (h_o * mask3d).sum(dim=1) / denom
                v_c = (h_c * mask3d).sum(dim=1) / denom
                per_layer.append(F.mse_loss(v_o, v_c))

            if per_layer:
                cf_inv_loss = torch.stack(per_layer).mean()

        if need_forget_hs:
            f_tok_mask_hit = f_tok_mask[hit_idx]  # [Bh,L]
            denom_b = f_tok_mask_hit.sum(dim=1, keepdim=True).clamp_min(1.0)
            mask3d = f_tok_mask_hit.unsqueeze(-1)  # [Bh,L,1]

            for hs_idx in self.target_layers:
                H_L = self._safe_fp32(Hf[hs_idx][hit_idx])  # [Bh,L,H]
                mu  = self._safe_fp32(getattr(self, f"safe_mu_L{hs_idx}").to(device))
                V   = self._safe_fp32(getattr(self, f"safe_basis_L{hs_idx}").to(device))

                z = (H_L * mask3d).sum(dim=1) / denom_b  # [Bh,H]

                proto = getattr(self, f"safe_proto_L{hs_idx}").to(device).float()
                proto_loss_list.append((z - proto.unsqueeze(0)).pow(2).mean())

                if self.use_info_bank and (self.info_repr_list is not None) and (self.w_info_energy > 0):
                    info_mu = getattr(self, f"info_mu_L{hs_idx}").to(device).float()
                    info_V  = getattr(self, f"info_basis_L{hs_idx}").to(device).float()
                    zc_info = z - info_mu.unsqueeze(0)
                    coeff = zc_info @ info_V
                    info_loss_list.append(coeff.pow(2).mean())

                zc = z - mu.unsqueeze(0)
                z_safe = (zc @ V) @ V.t()

                if self.basis_type.lower() == "safe":
                    z_danger = zc - z_safe
                else:
                    z_danger = z_safe
                    z_safe = zc - z_danger

                if self.alpha_energy > 0:
                    Hdim = float(z_danger.size(-1))
                    forget_eng_list.append((z_danger.pow(2).sum(dim=-1) / Hdim).mean())

                if self.w_dir > 0:
                    cos = F.cosine_similarity(zc, z_safe, dim=-1, eps=1e-6).mean()
                    forget_dir_list.append(1.0 - cos)

        forget_dir = torch.stack(forget_dir_list).mean() if forget_dir_list else torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        forget_eng = torch.stack(forget_eng_list).mean() if forget_eng_list else torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        proto_safe = torch.stack(proto_loss_list).mean() if proto_loss_list else torch.zeros((), device=device, dtype=torch.float32)
        info_energy = torch.stack(info_loss_list).mean() if info_loss_list else torch.zeros((), device=device, dtype=torch.float32)

        forget_weighted = self.w_dir * forget_dir + self.alpha_energy * forget_eng

        # refusal loss (hit-only)
        if self.w_refusal > 0 and has_forget and self.refusal_token_ids.numel() > 0:
            f_ids_hit = f_ids[hit_idx]
            f_mask_hit = f_mask[hit_idx]
            ref_ids = self.refusal_token_ids.to(device)
            ref_ids_batch = ref_ids.unsqueeze(0).expand(f_ids_hit.size(0), -1)
            ref_len = ref_ids_batch.size(1)
            ref_inputs = torch.cat([f_ids_hit, ref_ids_batch], dim=1)
            ref_mask = torch.cat(
                [f_mask_hit, torch.ones((f_ids_hit.size(0), ref_len), device=device, dtype=f_mask_hit.dtype)],
                dim=1
            )
            ref_labels = torch.full_like(ref_inputs, -100)
            ref_labels[:, -ref_len:] = ref_ids_batch
            out_ref = self.model(input_ids=ref_inputs, attention_mask=ref_mask, labels=ref_labels, return_dict=True)
            refusal_loss = out_ref.loss


        if self.w_forget_entropy > 0 and has_forget:

            s_logits = out_f.logits[hit_idx, :-1, :]
            win_mask = f_tok_mask[hit_idx, :-1]
            entropy_loss = -self._masked_entropy_from_logits(
                s_logits, win_mask, temperature=float(self.forget_entropy_temperature)
            )

        forget_unscaled = (
            forget_weighted
            + self.w_proto_safe * proto_safe
            + self.w_info_energy * info_energy
            + self.w_refusal * refusal_loss
            + self.w_forget_entropy * entropy_loss
            + self.w_cf_inv * cf_inv_loss
        )

        ramp = self._forget_ramp()


        forget_scale = 1.0
        if self.forget_hit_boost and has_forget:
            Bh = float(hit_idx.numel())
            forget_scale = min(float(self.forget_hit_boost_max), float(Bf) / max(1.0, Bh))
        self.log("sched/forget_hit_boost", torch.tensor(forget_scale, device=device), on_step=True, prog_bar=False)

        forget_weighted = forget_unscaled * ramp * float(forget_scale)
        self.log("sched/forget_ramp", torch.tensor(ramp, device=device), on_step=True, prog_bar=False)
        self.log("loss/forget_unscaled", forget_unscaled.detach(), on_step=True, prog_bar=False)
        self.log("loss/forget_scaled", forget_weighted.detach(), on_step=True, prog_bar=False)
        if self.w_refusal > 0:
            self.log("loss/refusal", (self.w_refusal * refusal_loss).detach(), on_step=True, prog_bar=False)


        keep_loss = torch.zeros((), device=device, dtype=torch.float32)
        if (self.lambda_bg_keep > 0.0) and (self.teacher is not None):

            bg_mask = (f_mask.float() - f_tok_mask).clamp_min(0.0)
            if self.bg_keep_on_all_forget:
                if bg_mask.sum() > 0:
                    with torch.no_grad():
                        t_out_f = self.teacher(input_ids=f_ids, attention_mask=f_mask, return_dict=True)
                    keep_loss = self._masked_kl(
                        out_f.logits[:, :-1, :],
                        t_out_f.logits[:, :-1, :],
                        bg_mask[:, :-1],
                        temperature=float(self.bg_keep_temperature),
                    )
            elif has_forget:
                bg_mask_hit = bg_mask[hit_idx]
                if bg_mask_hit.sum() > 0:
                    with torch.no_grad():
                        t_out_f = self.teacher(input_ids=f_ids[hit_idx], attention_mask=f_mask[hit_idx], return_dict=True)
                    keep_loss = self._masked_kl(
                        out_f.logits[hit_idx, :-1, :],
                        t_out_f.logits[:, :-1, :],
                        bg_mask_hit[:, :-1],
                        temperature=float(self.bg_keep_temperature),
                    )

        r_ids_cpu = batch["retain_input_ids"]
        r_mask_cpu = batch["retain_attention_mask"]
        Br, Lr = r_ids_cpu.size()

        rtn_hits = self._find_hits_cpu(
            r_ids_cpu,
            self.retain_name_token_ids,
            Lr,
            allow_fuzzy_match=self.allow_fuzzy_match_retain,
            max_name_span=self.max_name_span_retain,
        )

        r_ids  = r_ids_cpu.to(device, non_blocking=True)
        r_mask = r_mask_cpu.to(device, non_blocking=True)
        need_retain_hs = self.w_retain_hs > 0
        out_r = self.model(input_ids=r_ids, attention_mask=r_mask,
                           output_hidden_states=need_retain_hs, return_dict=True)
        Hr = tuple(h.float() for h in out_r.hidden_states) if need_retain_hs else None
        r_tok_mask = self._window_mask_from_hits(
            rtn_hits,
            int(max(0, self.retain_window)),
            r_mask,
            include_hit=self.hit_include_token,
            both_sides=self.hit_both_sides,
            decay=0.0,  # keep retain window hard by default
        ).float()

        self.log("dbg/num_retain_hits", torch.tensor(sum(len(h) for h in rtn_hits), device=device), prog_bar=True)
        self.log("dbg/retain_tokmask_mean", r_tok_mask.mean(), prog_bar=False)
        self.log("dbg/retain_tokmask_sum", r_tok_mask.sum(), prog_bar=True)
        self.log("dbg/retain_nonempty_seqs", (r_tok_mask.sum(dim=1) > 0).float().sum(), prog_bar=True)

        retain_weighted = torch.zeros((), device=device, dtype=torch.float32)
        if (self.teacher is not None) and self.w_retain_align > 0:
            with torch.no_grad():
                t_out_r = self.teacher(input_ids=r_ids, attention_mask=r_mask,
                                       output_hidden_states=need_retain_hs, return_dict=True)
            retain_kl = self._masked_kl(out_r.logits, t_out_r.logits, r_mask, temperature=self.retain_kl_temperature)
            retain_weighted = self.w_retain_align * retain_kl

            if need_retain_hs and r_tok_mask.sum() > 0:
                Ht_r = tuple(h.float() for h in t_out_r.hidden_states)
                denom_r = (r_tok_mask.sum() * Hr[1].size(-1)).clamp_min(torch.tensor(1.0, device=device, dtype=torch.float32))
                mask3d_r = r_tok_mask.unsqueeze(-1)
                per_layer_r = []
                for hs_idx in self.target_layers:
                    diff = (Hr[hs_idx] - Ht_r[hs_idx]) * mask3d_r
                    per_layer_r.append(self._nanfix(diff.pow(2).sum()) / denom_r)
                retain_mse = torch.stack(per_layer_r).mean()
                retain_weighted = retain_weighted + (self.w_retain_hs * retain_mse)


        if (self.forget_end_step >= 0) and (int(self.global_step) >= self.forget_end_step):
            forget_weighted = torch.zeros((), device=device, dtype=torch.float32)

        if (self.retain_only_steps > 0) and (self.global_step < self.retain_only_steps):
            forget_weighted = torch.zeros((), device=device, dtype=torch.float32)
            keep_loss = torch.zeros((), device=device, dtype=torch.float32)

        lora_reg = torch.zeros((), device=device, dtype=torch.float32)
        if self.w_lora_reg > 0:
            lora_reg = self._lora_l2()

        final_loss = forget_weighted + self.lambda_bg_keep * keep_loss + retain_weighted + self.w_lora_reg * lora_reg

        self.log("train_loss", final_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("loss/forget_total", forget_weighted.detach(), on_step=True, prog_bar=True)
        self.log("loss/retain_total", retain_weighted.detach(), on_step=True, prog_bar=True)
        if self.lambda_bg_keep > 0:
            self.log("loss/bg_keep", keep_loss.detach(), on_step=True, prog_bar=False)
        if self.w_lora_reg > 0:
            self.log("loss/lora_reg", (self.w_lora_reg * lora_reg).detach(), on_step=True, prog_bar=False)

        self.log("forget/raw_dir", forget_dir.detach(), on_step=True, prog_bar=False)
        self.log("forget/raw_energy", forget_eng.detach(), on_step=True, prog_bar=False)
        self.log("forget/proto_safe", proto_safe.detach(), on_step=True, prog_bar=False)
        if self.use_info_bank and (self.info_repr_list is not None):
            self.log("forget/info_energy", info_energy.detach(), on_step=True, prog_bar=False)
        if self.w_forget_entropy > 0:
            self.log("forget/entropy_loss", (self.w_forget_entropy * entropy_loss).detach(), on_step=True, prog_bar=False)
        if self.w_cf_inv > 0:
            self.log("forget/cf_inv_loss", (self.w_cf_inv * cf_inv_loss).detach(), on_step=True, prog_bar=False)

        if batch_idx == 0:
            print(
                f"[dbg] ramp={ramp:.4f} boost={forget_scale:.3f} "
                f"forget_unscaled={float(forget_unscaled.detach()):.6f} "
                f"forget_scaled={float(forget_weighted.detach()):.6f} "
                f"bg_keep={float(keep_loss.detach()):.6f} "
                f"retain_total={float(retain_weighted.detach()):.6f} "
                f"lora_reg={float(lora_reg.detach()):.6f}"
            )

        return final_loss

    def on_fit_start(self):
        if self.teacher is None:
            return
        self._run_teacher_validation()

    def _run_teacher_validation(self):
        loaders = self.val_dataloader()
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        device = next(self.teacher.parameters()).device
        self.teacher.eval()

        total_seqacc = 0.0
        total_entropy = 0.0
        total_conf_gap = 0.0
        total_low_conf = 0.0
        total_batches = 0

        with torch.no_grad():
            for loader_idx, loader in enumerate(loaders):
                loader_seqacc = 0.0
                loader_entropy = 0.0
                loader_conf_gap = 0.0
                loader_low_conf = 0.0
                loader_batches = 0
                for batch in loader:
                    seqacc, entropy, conf_gap, low_conf = self._teacher_validation_seq2seq(batch, device)
                    total_seqacc += seqacc
                    total_entropy += entropy
                    total_conf_gap += conf_gap
                    total_low_conf += low_conf
                    total_batches += 1
                    loader_seqacc += seqacc
                    loader_entropy += entropy
                    loader_conf_gap += conf_gap
                    loader_low_conf += low_conf
                    loader_batches += 1
                if loader_batches > 0:
                    print(
                        f"[teacher_val][loader={loader_idx}] "
                        f"seqacc={loader_seqacc / loader_batches:.6f} "
                        f"entropy={loader_entropy / loader_batches:.6f} "
                        f"conf_gap={loader_conf_gap / loader_batches:.6f} "
                        f"low_conf_ratio={loader_low_conf / loader_batches:.6f}"
                    )

        if total_batches > 0:
            avg_seqacc = total_seqacc / total_batches
            avg_entropy = total_entropy / total_batches
            avg_conf_gap = total_conf_gap / total_batches
            avg_low_conf = total_low_conf / total_batches
            if self.logger is not None and hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {
                        "teacher/val_seqacc": avg_seqacc,
                        "teacher/val_entropy": avg_entropy,
                        "teacher/val_conf_gap": avg_conf_gap,
                        "teacher/val_low_conf_ratio": avg_low_conf,
                    }
                )
            print(
                f"[teacher_val] seqacc={avg_seqacc:.6f} "
                f"entropy={avg_entropy:.6f} "
                f"conf_gap={avg_conf_gap:.6f} "
                f"low_conf_ratio={avg_low_conf:.6f}"
            )

    def _teacher_validation_seq2seq(self, batch, device):
        input_ids = batch['source_ids'].to(device)
        attention_mask = batch['source_mask'].to(device)
        labels = batch["target_ids"].to(device)

        generated = self.teacher.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_legacy_cache=True,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        generated_ids = generated.sequences[:, input_ids.size(1):]
        decoded_preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]
        scores = [self.similarity_score(gen, true) for gen, true in zip(decoded_preds, decoded_labels)]
        average_score = sum(scores) / len(scores)

        logits = self.teacher(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
        entropy_scores = self.token_level_entropy(logits)
        conf_gap_scores = self.confidence_gap(logits)
        low_conf_ratio_scores = self.low_confidence_ratio(logits, threshold=0.7)

        return (
            average_score,
            float(entropy_scores.mean().detach().cpu()),
            float(conf_gap_scores.mean().detach().cpu()),
            float(low_conf_ratio_scores.mean().detach().cpu()),
        )

    def _is_last_epoch(self):
        if getattr(self, "trainer", None) is None:
            return False
        max_epochs = getattr(self.trainer, "max_epochs", None)
        if max_epochs is None or max_epochs < 0:
            return False
        return (int(self.trainer.current_epoch) + 1) >= int(max_epochs)

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        #if not self._is_last_epoch():
        #    return None
        if self.mode == 'unlearn':
            return self.validation_seq2seq(batch, dataloader_idx=dataloader_idx)
        else:
            raise Exception(f'Currently not supporting {self.mode}')

    def validation_seq2seq(self, batch, dataloader_idx=-1):
        input_ids = batch['source_ids']
        attention_mask = batch['source_mask']
        labels = batch["target_ids"]

        if not self._printed_val_prompt:
            print("val_prompt_sample:", self.tokenizer.decode(input_ids[0], skip_special_tokens=False))
            self._printed_val_prompt = True

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_legacy_cache=True,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        generated_ids = generated.sequences[:, input_ids.size(1):]
        decoded_preds  = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]
        scores = [self.similarity_score(gen, true) for gen, true in zip(decoded_preds, decoded_labels)]
        average_score = sum(scores) / len(scores)

        self.log('val_seqacc', average_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if isinstance(dataloader_idx, int) and (dataloader_idx >= 0) and (dataloader_idx < len(self._val_type_by_idx)):
            vtype = self._val_type_by_idx[dataloader_idx]
        else:
            vtype = f"dl{dataloader_idx}"
        self.log(
            f"val_seqacc/{vtype}",
            average_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        print("decoded_preds:", decoded_preds[:20])
        print("decoded_labels:", decoded_labels[:20])
        print(f"Average Similarity Score: {average_score * 100:.2f}%")

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
            entropy_scores = self.token_level_entropy(logits)
            conf_gap_scores = self.confidence_gap(logits)
            low_conf_ratio_scores = self.low_confidence_ratio(logits, threshold=0.7)
            self.log("val_entropy", entropy_scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_conf_gap", conf_gap_scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_low_conf_ratio", low_conf_ratio_scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'val_acc': average_score, 'decoded_preds': decoded_preds, 'decoded_labels': decoded_labels}


    def similarity_score(self, generated_answer, true_answer):
        return SequenceMatcher(None, generated_answer.lower().strip(), true_answer.lower().strip()).ratio()

    def token_level_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean(dim=-1)

    def confidence_gap(self, logits):
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        return max_probs.mean(dim=-1)

    def low_confidence_ratio(self, logits, threshold=0.7):
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        low_conf_mask = (max_probs < threshold).float()
        return low_conf_mask.mean(dim=-1)

    def get_dataset(self, dataset_name, tokenizer, valid_subset_path, type_path, length=None):
        input_length = self.hparams.input_length if length is None else length
        output_length = length if length else self.hparams.output_length
        from Datasets import Custom_Dataset
        dataset = Custom_Dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            valid_subset_path=valid_subset_path,
            type_path=type_path,
            input_length=input_length,
            output_length=output_length,
            args=self.hparams
        )
        return dataset

    def train_dataloader(self):
        dataset = self.hparams.train_set
        train_dataset = self.get_dataset(dataset, self.tokenizer, "", "train", length=None)
        sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset, sampler=sampler,
                          batch_size=self.hparams.train_batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        datasets, target_idx = [], -1
        for i in range(len(self.hparams.valid_sets)):
            ds = self.get_dataset(
                dataset_name=self.hparams.valid_sets[i],
                tokenizer=self.tokenizer,
                valid_subset_path=self.hparams.valid_subset_path[i],
                type_path=self.hparams.valid_type_path[i],
                length=None
            )
            datasets.append(ds)
        if self.mode in ['unlearn'] and 'target' in self.hparams.valid_type_path:
            target_idx = self.hparams.valid_type_path.index('target')
        self._val_type_by_idx = list(self.hparams.valid_type_path)

        loaders = []
        for i, ds in enumerate(datasets):
            if self.mode in ['unlearn'] and i == target_idx:
                bs = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps
                loaders.append(DataLoader(ds, batch_size=bs, num_workers=self.hparams.num_workers, shuffle=False))
            elif self.mode in ['unlearn'] and i == 1:
                bs = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps
                loaders.append(DataLoader(ds, batch_size=bs, num_workers=self.hparams.num_workers, shuffle=False))
            else:
                loaders.append(DataLoader(ds, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False))
        return loaders

    def on_validation_epoch_end(self):
        #if not self._is_last_epoch():
            #return
        if getattr(self.trainer, "sanity_checking", False):
            return
        if self.early_stop_target_acc < 0 or self.early_stop_test_acc < 0:
            return
        if not self._val_type_by_idx:
            return

        tgt_key = f"val_seqacc/{self.early_stop_target_type}"
        test_key = f"val_seqacc/{self.early_stop_test_type}"
        metrics = getattr(self.trainer, "callback_metrics", {})
        tgt = metrics.get(tgt_key, None)
        test = metrics.get(test_key, None)

        if (tgt is None) and (self.early_stop_target_type in self._val_type_by_idx):
            tgt_idx = self._val_type_by_idx.index(self.early_stop_target_type)
            tgt = metrics.get(f"{tgt_key}/dataloader_idx_{tgt_idx}", None)
            if tgt is None:
                tgt = metrics.get(f"val_seqacc_step/dataloader_idx_{tgt_idx}", None)
        if (test is None) and (self.early_stop_test_type in self._val_type_by_idx):
            test_idx = self._val_type_by_idx.index(self.early_stop_test_type)
            test = metrics.get(f"{test_key}/dataloader_idx_{test_idx}", None)
            if test is None:
                test = metrics.get(f"val_seqacc_step/dataloader_idx_{test_idx}", None)
        if tgt is None or test is None:
            return

        tgt_v = float(tgt.detach().cpu()) if torch.is_tensor(tgt) else float(tgt)
        test_v = float(test.detach().cpu()) if torch.is_tensor(test) else float(test)

        if (tgt_v < self.early_stop_target_acc) and (test_v > self.early_stop_test_acc):
            self._early_stop_ok_epochs += 1
        else:
            self._early_stop_ok_epochs = 0

        if self._early_stop_ok_epochs >= max(1, int(self.early_stop_patience)):
            self.log("early_stop/triggered", torch.tensor(1.0, device=self.device), on_step=False, on_epoch=True, prog_bar=True)
            self.trainer.should_stop = True

    def export_hf_checkpoint(self, output_dir: str, merge_lora: bool = True, dtype: str = "bf16"):

        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model

        if merge_lora and hasattr(model_to_save, "merge_and_unload"):
            model_to_save = model_to_save.merge_and_unload()

        if dtype.lower() in ("bf16", "bfloat16"):
            model_to_save = model_to_save.to(torch.bfloat16)
        elif dtype.lower() in ("fp16", "float16"):
            model_to_save = model_to_save.to(torch.float16)
        else:
            model_to_save = model_to_save.to(torch.float32)

        model_to_save = model_to_save.to("cpu")

        model_to_save.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)


        with open(os.path.join(output_dir, "EXPORT_INFO.txt"), "w", encoding="utf-8") as f:
            f.write(f"exported_from={self.hparams.model_name_or_path}\n")
            f.write(f"merge_lora={merge_lora}\n")
            f.write(f"dtype={dtype}\n")

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def device(self):
        return self._device
