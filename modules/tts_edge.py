# modules/tts_edge.py
import asyncio, uuid, os, numpy as np, librosa, soundfile as sf, edge_tts, json
from pathlib import Path

def _trim(y, top_db=40): 
    yt, _ = librosa.effects.trim(y, top_db=top_db); 
    return yt

def _fade(y, sr, ms=12):
    n, f = len(y), int(ms*sr/1000)
    if f <= 0 or n <= 2*f: return y
    r = np.linspace(0.0, 1.0, f, dtype=np.float32)
    y[:f] *= r; y[-f:] *= r[::-1]; return y

def _micro_stretch(y, sr, target_sec, cap=0.12):
    cur = max(1e-6, len(y)/sr)
    if target_sec <= 0: return np.zeros(0, dtype=np.float32)
    ratio = cur/target_sec
    if abs(1.0 - ratio) <= cap:
        return librosa.effects.time_stretch(y, rate=ratio)
    return y

async def _edge_say_to_array(text, voice, rate_pct=0, sr=24000, retries=3):
    if not text.strip(): return np.zeros(0, dtype=np.float32)
    err = None
    for k in range(1, retries+1):
        try:
            tmp = f"_edge_{uuid.uuid4().hex}.mp3"
            await edge_tts.Communicate(text=text, voice=voice, rate=f"{int(rate_pct):+d}%").save(tmp)
            y, _ = librosa.load(tmp, sr=sr, mono=True); os.remove(tmp)
            return y.astype(np.float32)
        except Exception as e:
            err = e; await asyncio.sleep(0.3*k)
    raise err

async def _synth_exact(text, voice, target_sec, sr_synth=24000, sr_out=16000, rate_cap=30):
    if not text.strip() or target_sec <= 0:
        return np.zeros(int(target_sec*sr_out), dtype=np.float32)
    y1 = await _edge_say_to_array(text, voice, 0, sr_synth)
    y1 = _trim(y1); d1 = max(1e-6, len(y1)/sr_synth)
    want = int(np.clip((1.0/(d1/target_sec) - 1.0)*100.0, -rate_cap, +rate_cap))
    y2 = await _edge_say_to_array(text, voice, want, sr_synth)
    y2 = _trim(y2)
    y2 = _micro_stretch(y2, sr_synth, target_sec, cap=0.12)
    if sr_synth != sr_out:
        y2 = librosa.resample(y2, orig_sr=sr_synth, target_sr=sr_out)
    tgt_n = int(round(target_sec*sr_out))
    if len(y2) > tgt_n: y2 = y2[:tgt_n]
    if len(y2) < tgt_n: y2 = np.pad(y2, (0, tgt_n - len(y2)))
    y2 = _fade(y2, sr_out, ms=12)
    pk = float(np.max(np.abs(y2)) + 1e-9)
    if pk > 0: y2 = (y2 / pk) * 0.9
    return y2

async def pick_edge_voice(locale_prefix: str, preferred: str|None):
    voices = await edge_tts.list_voices()
    if preferred:
        for v in voices:
            if v.get("ShortName","").lower() == preferred.lower():
                print(f"[TTS] Using voice: {v['ShortName']}")
                return v["ShortName"]
        print(f"[TTS] Preferred '{preferred}' not found, auto-picking…")
    cand = [v for v in voices if v.get("Locale","").lower().startswith(locale_prefix.lower())]
    if not cand: raise RuntimeError(f"No voices for locale prefix '{locale_prefix}'")
    # Prefer Neural + female
    cand.sort(key=lambda v: (("Neural" not in v.get("ShortName","")), v.get("Gender","")!="Female"))
    print(f"[TTS] Using voice: {cand[0]['ShortName']}")
    return cand[0]["ShortName"]

def _place_overwrite(dst, start_idx, seg):
    s = int(start_idx); e = min(len(dst), s + len(seg))
    if s >= len(dst) or e <= 0: return
    seg_slice = seg[max(0, -s): max(0, -s) + (e - max(0, s))]
    s = max(0, s); e = s + len(seg_slice)
    dst[s:e] = seg_slice

def _is_punct_end(txt): return txt.strip().endswith((".", "?", "!"))

async def build_dubbed_timeline(
    asr_json: Path, trans_json: Path, orig_audio_wav: Path,
    out_wav: Path, locale_prefix: str, preferred_voice: str|None,
    sr_synth=24000, sr_out=16000,
    gap_split=0.35, left_borrow=0.40, right_borrow=0.60, borrow_frac=0.85
):
    import json
    with open(asr_json, "r", encoding="utf-8") as f: asr = json.load(f)
    with open(trans_json, "r", encoding="utf-8") as f: trs = json.load(f)
    assert len(asr) == len(trs), "ASR/translation counts differ"

    orig, _ = librosa.load(str(orig_audio_wav), sr=sr_out, mono=True)
    timeline = np.zeros_like(orig, dtype=np.float32)

    # group segments (sentence-ish)
    groups, cur = [], [0]
    for i in range(1, len(asr)):
        gap = float(asr[i]["start"]) - float(asr[i-1]["end"])
        if (gap > gap_split) or _is_punct_end(asr[i-1]["text"]):
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)

    # helpers
    def base_window(g):
        return float(asr[g[0]]["start"]), float(asr[g[-1]]["end"])
    def prev_end(idx):  return float(asr[groups[idx-1][-1]]["end"]) if idx>0 else 0.0
    def next_start(idx):return float(asr[groups[idx+1][0]]["start"]) if idx<len(groups)-1 else len(orig)/sr_out

    voice = await pick_edge_voice(locale_prefix, preferred_voice)

    for gi, g in enumerate(groups, 1):
        s0, e0 = base_window(g)
        left  = max(0.0, s0 - prev_end(gi-1))
        right = max(0.0, next_start(gi-1) - e0)
        borrowL = min(left_borrow,  left  * borrow_frac)
        borrowR = min(right_borrow, right * borrow_frac)
        S = s0 - borrowL; E = e0 + borrowR; target = max(0.10, E - S)
        text = " ".join((trs[i]["tgt"] or "").strip() for i in g).strip()
        y = await _synth_exact(text, voice, target_sec=target, sr_synth=sr_synth, sr_out=sr_out)
        _place_overwrite(timeline, int(round(S*sr_out)), y)
        print(f"[TTS] Group {gi}/{len(groups)} [{S:.2f}-{E:.2f}] {len(g)} segs")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, timeline.astype(np.float32), sr_out)
    print("✅ TTS dubbed wav:", out_wav)
    return out_wav
