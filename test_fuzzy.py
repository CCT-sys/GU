import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PT = "fuzzy_repr-8b.pt"  
NUM_FUZZY_NEEDED = 20
IDK_PROMPTS_PATH = "idk_prompts.jsonl"
REJECTS_PATH = "non_idk_samples.jsonl"
SAVE_REJECTS = 200

FUZZY_KEYWORDS = [
    "i don't know", "i am not sure", "i'm not sure",
    "i cannot answer", "i can't answer", "no idea",
    "i'm unsure", "uncertain", "unknown",
    "i'm sorry", "sorry", "i cannot help", "i can't help",
    "i cannot provide", "i can't provide", "unable to",
    "privacy", "personal information", "private information",
]

NAMES = []
TEMPLATES = [

]
PROMPTS = [tpl.format(name) for name in NAMES for tpl in TEMPLATES]

MODEL_PATH = ""
CACHE_DIR = ""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, output_hidden_states=True).to(DEVICE).eval()

def find_name_last_positions(input_ids_1d, name, tokenizer, max_span=8):

    name_ids = tokenizer.encode(name, add_special_tokens=False)
    name_ids = name_ids[:max_span]
    if not name_ids: 
        return []
    ids = input_ids_1d.tolist()
    hits = []

    for j in range(0, len(ids) - len(name_ids) + 1):
        window = ids[j:j+len(name_ids)]
        same = sum(int(a == b) for a, b in zip(window, name_ids))
        if same >= max(len(name_ids) - 1, int(0.8 * len(name_ids))):
            hits.append(j + len(name_ids) - 1)
    return sorted(set(hits))

def collect_idk_safe_repr_for_prompt(prompt, W=16):

    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=50, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    completion_ids = gen.sequences[0, prompt_len:]
    resp = tokenizer.decode(completion_ids, skip_special_tokens=True).lower().strip()
    if not any(k in resp for k in FUZZY_KEYWORDS):
        return None  
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  
    L = hs[0].size(1)


    mask = torch.zeros(L, dtype=torch.float32, device=DEVICE)
    for name in NAMES:
        last_pos_list = find_name_last_positions(enc["input_ids"][0], name, tokenizer)
        for p in last_pos_list:
    
            left = max(0, p - W)
            right = min(L - 1, p + W)
            mask[left:right+1] = 1.0
            mask[p] = 1.0
    if mask.sum() == 0:
        return None  

    mask = mask.view(1, L, 1)  # [1,L,1]
    layer_vecs = []
    for layer in hs:  # [1,L,H]
 
        num = mask.sum().clamp_min(1.0)
        v = (layer * mask).sum(dim=1) / num  # [1,H]
        layer_vecs.append(v.squeeze(0).detach().cpu())
    return torch.stack(layer_vecs)  # [num_layers, H]


fuzzy_reprs = []
idk_prompts = []
rejects = []
for prompt in tqdm(PROMPTS):
    vec = collect_idk_safe_repr_for_prompt(prompt, W=16)
    if vec is not None:
        fuzzy_reprs.append(vec)
        idk_prompts.append({"prompt": prompt})
    else:
        if len(rejects) < SAVE_REJECTS:
            rejects.append({"prompt": prompt})
    if len(fuzzy_reprs) >= NUM_FUZZY_NEEDED:
        break

if fuzzy_reprs:
    torch.save(fuzzy_reprs, OUTPUT_PT)
    print(f"Saved {len(fuzzy_reprs)} samples to {OUTPUT_PT}")
    if idk_prompts:
        with open(IDK_PROMPTS_PATH, "w", encoding="utf-8") as f:
            for rec in idk_prompts:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(idk_prompts)} IDK prompts to {IDK_PROMPTS_PATH}")
    if rejects:
        with open(REJECTS_PATH, "w", encoding="utf-8") as f:
            for rec in rejects:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(rejects)} non-IDK prompts to {REJECTS_PATH}")
else:
    print("No IDK anchors collected.")
