# modules/translate_nllb.py
from pathlib import Path
import json, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "facebook/nllb-200-distilled-600M"

def _get_bos_id(tok, lang_code: str) -> int:
    if hasattr(tok, "lang_code_to_id") and isinstance(tok.lang_code_to_id, dict):
        if lang_code in tok.lang_code_to_id:
            return tok.lang_code_to_id[lang_code]
    if hasattr(tok, "get_lang_id"):
        return tok.get_lang_id(lang_code)
    bid = tok.convert_tokens_to_ids(lang_code)
    if isinstance(bid, int) and bid != tok.unk_token_id:
        return bid
    raise ValueError(f"Could not resolve BOS id for {lang_code}")

class NLLBTranslator:
    def __init__(self, device=None):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.device = device

    def translate_texts(self, texts, tgt_code, beams=4, max_new=200, batch_size=8):
        out = []
        bos_id = _get_bos_id(self.tok, tgt_code)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = self.model.generate(
                **enc, forced_bos_token_id=bos_id, num_beams=beams,
                max_new_tokens=max_new, no_repeat_ngram_size=3
            )
            outs = self.tok.batch_decode(gen, skip_special_tokens=True)
            out.extend(outs)
        return out

def translate_segments(asr_json: Path, tgt_code: str, out_json: Path):
    with open(asr_json, "r", encoding="utf-8") as f:
        segs = json.load(f)
    texts = [s["text"] for s in segs]
    tr = NLLBTranslator()
    trans = tr.translate_texts(texts, tgt_code)
    out = []
    for s, t in zip(segs, trans):
        out.append({"start": float(s["start"]), "end": float(s["end"]), "src": s["text"], "tgt": t})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("✅ Translation saved:", out_json)
    return out_json
