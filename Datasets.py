import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
import re
from utils import normalize_reply, DIALOG_DATASETS
import spacy
import json

from json import JSONDecodeError

class Custom_Dataset(Dataset):
    def __init__(self, tokenizer, dataset_name, valid_subset_path, type_path,
                 input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path
        self.valid_subset_path = valid_subset_path

        def load_local_file_to_df(path: str) -> pd.DataFrame:
            path_lower = path.lower()
            if path_lower.endswith(".csv"):
                df = pd.read_csv(path, lineterminator="\n", on_bad_lines="skip", encoding="utf-8")
                return df

            if path_lower.endswith(".json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)  # 标准 JSON：list/dict
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    elif isinstance(data, dict):
                        return pd.DataFrame([data])
                    else:
                        raise ValueError(f"Unsupported JSON root type: {type(data)}")
                except JSONDecodeError:
                    return pd.read_json(path, lines=True, encoding="utf-8")


            if path_lower.endswith(".jsonl"):
                df = pd.read_json(path, lines=True, encoding="utf-8")
                return df

            raise ValueError(f"Unsupported local file type: {path}")

        if isinstance(self.dataset_name, str) and (
            self.dataset_name.lower().endswith((".csv", ".json", ".jsonl"))
        ):
            self.dataset = load_local_file_to_df(self.dataset_name)
            self.dataset.rename(columns=lambda x: x.strip(), inplace=True)

            if "answer" in self.dataset.columns:
                self.dataset["answer"] = self.dataset["answer"].fillna("")
            if "original_answer" in self.dataset.columns:
                self.dataset["original_answer"] = self.dataset["original_answer"].fillna("")

        else:
            if valid_subset_path:
                dataset = load_dataset(
                    self.dataset_name,
                    valid_subset_path,
                    split=type_path,
                    ignore_verifications=True,
                    cache_dir=args.cache_dir
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=type_path,
                    ignore_verifications=True,
                    cache_dir=args.cache_dir
                )
            self.dataset = dataset.to_pandas()


        if self.type_path == "train":

            target_q=""
    

            q = self.dataset["question"].astype(str).str.strip()
            hit = self.dataset.index[q.eq(target_q.strip())]


            if len(hit) == 0:
                raise ValueError("Target question not found; cannot split forget/retain by cutoff.")

            else:
                cutoff = int(hit.max())  
            print("------------",cutoff)

            self.forget_dataset = self.dataset.iloc[:cutoff + 1].reset_index(drop=True)
   
            self.retain_dataset = self.dataset.iloc[cutoff + 1:].reset_index(drop=True)

            if len(self.retain_dataset) == 0:
                raise ValueError("retain_dataset is empty after split; choose an earlier cutoff.")


            self.length = max(len(self.forget_dataset), len(self.retain_dataset))


            print(f"[Split] cutoff={cutoff}, forget={len(self.forget_dataset)}, retain={len(self.retain_dataset)}")
            print("[Split] forget last Q:", self.forget_dataset.iloc[-1]["question"][:120])
            print("[Split] retain first Q:", self.retain_dataset.iloc[0]["question"][:120])

        self.length = len(self.dataset)
    def __len__(self):
        return self.length


    def convert_to_features(self, example_batch):

        tokenized = self.tokenizer(example_batch['question'], max_length=self.input_length, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()

        
        if 'original_answer' in example_batch and example_batch['original_answer']:
            target_text = example_batch['original_answer']
        elif 'answer' in example_batch and example_batch['answer']:
            target_text = example_batch['answer']
        else:
            target_text = None

        if target_text:
            tokenized_target = self.tokenizer(
            target_text,
            max_length=self.output_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
            )
            target_ids = tokenized_target.input_ids.squeeze()
        else:
            target_ids = None

        return input_ids, attention_mask, target_ids

    def __getitem__(self, index):
        if self.type_path == "train":

            

            row_forget = self.forget_dataset.iloc[index % len(self.forget_dataset)]
            row_retain = self.retain_dataset.iloc[index % len(self.retain_dataset)]

            forget_text = f"{row_forget['question']}"
            retain_text = f"{row_retain['question']}"

            forget_in = self.tokenizer(
                forget_text,
                truncation=True,
                padding='max_length',        
                max_length=self.input_length,
                return_tensors="pt"
            )

            forget_input_ids = forget_in["input_ids"].squeeze(0)
            forget_attention_mask = forget_in["attention_mask"].squeeze(0)
            forget_labels = forget_input_ids.clone()

            retain_in = self.tokenizer(
                retain_text,
                truncation=True,
                padding='max_length',        
                max_length=self.input_length,
                return_tensors="pt"
            )
            retain_input_ids = retain_in["input_ids"].squeeze(0)
            retain_attention_mask = retain_in["attention_mask"].squeeze(0)
            retain_labels = retain_input_ids.clone()


            out = {
                "forget_input_ids": forget_input_ids,
                "forget_attention_mask": forget_attention_mask,
                "forget_labels": forget_labels,
                "retain_input_ids": retain_input_ids,
                "retain_attention_mask": retain_attention_mask,
                "retain_labels": retain_labels                
            }
            if bool(getattr(self.args, "debug_hits", False)):
                out["forget_text"] = forget_text
                out["retain_text"] = retain_text
            return out

        else:
            data = self.dataset.iloc[index]
            input_ids, attention_mask, target_ids = self.convert_to_features(data)
            return {
                "source_ids": input_ids,
                "source_mask": attention_mask,
                "target_ids": target_ids

            }
