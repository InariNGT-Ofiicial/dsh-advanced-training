#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_cer.py —— 无参考验收 A/B：ASR 回转写 + CER + 末尾字覆盖 + 尾音统计。

设计立场（三条模型线通用）：
  1. 度量与模型线无关。不管 A/B 的产物是 GPT-SoVITS 合成、DDSP 换声还是 RVC 换声，
     落到磁盘上都是 wav——同一把尺子量完才能同台对比。
  2. 「无参考」= 不需要人工评分；参考文本仍要给（测试集本来就是带台词设计的），
     CER 才有分母。没有 refs 时退化为只报 ASR 转写 + 尾音统计。
  3. ★ ASR 也是会犯错的第三方：CER 差不全是模型的锅（同音字、人名、数字读法）。
     看**相对差**（A vs B 同一批句子）与**末尾字覆盖**这种结构性指标，
     别拿绝对 CER 单独下结论。

末尾字覆盖（tail_hit）口径：
  ref 去掉标点/空白后取最后 k 个字符（默认 3，即「末三字」），
  在 asr 去掉标点/空白后的最后 8 个字符里找子串；找到即 hit。
  这是「留尾修复」的直接验收指标：切尾训出来的模型会吞末字，
  实测末三字念出率 33%（旧出料）→ 93%（留尾出料）。

依赖：faster-whisper（GPT-SoVITS venv 里现成有）。用 GSV 的 python 跑本脚本。

用法：
  python asr_cer.py --dir <wav目录> --refs refs.json
  python asr_cer.py --dir A --dir-b B --refs refs.json        # A/B 对比
  python asr_cer.py --dir ... --limit 20                      # 抽样
refs.json 格式：{"<文件名或去扩展名>": "参考台词", ...}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_audit import end_metrics, scan_audio          # noqa: E402

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print(json.dumps({"ok": False, "error": "缺少 soundfile/numpy"}, ensure_ascii=False))
    sys.exit(2)

# 去掉这些字符再比（中日英标点 + 空白）。ASR 与 ref 两边同规则，公平。
_PUNCT = re.compile(r"[\s，，。．、！？!?\.，,:：;；「」『』“”\"'（）()\[\]【】·…—～\-―]+")

try:
    import faster_whisper                               # noqa: F401
except ImportError:
    print(json.dumps({"ok": False,
                      "error": "缺少 faster_whisper（用 GPT-SoVITS venv 的 python 跑）"},
                     ensure_ascii=False))
    sys.exit(2)


def ensure_cuda_dlls():
    """
    ctranslate2 找 cublas64_12.dll 只看系统 PATH；从插件独立拉起的进程里，
    torch 的 lib 目录不在 PATH，会报 'Library cublas64_12.dll is not found'
    （GSV 自家脚本不踩这个坑，是因为 webui 启动时把 venv 环境整个带进来了）。
    这里按「当前解释器的 venv → site-packages/torch/lib」自愈。
    """
    venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    for d in (os.path.join(venv, "Lib", "site-packages", "torch", "lib"),
              os.path.join(venv, "Lib", "site-packages", "ctranslate2")):
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


ensure_cuda_dlls()


def load_refs(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # key 同时收「文件名」与「去扩展名」两种形态，查的时候不用猜
    out = {}
    for k, v in raw.items():
        out[k] = v
        out[os.path.splitext(k)[0]] = v
    return out


def cer(ref: str, hyp: str) -> float:
    """字符级编辑距离 / 参考长度。ref 为空时返回 None。"""
    r = _PUNCT.sub("", ref)
    h = _PUNCT.sub("", hyp)
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if rc == hc else 1))
        prev = cur
    return round(prev[len(h)] / len(r), 3)


def tail_hit(ref: str, hyp: str, k: int = 3, window: int = 8) -> bool:
    r = _PUNCT.sub("", ref)
    h = _PUNCT.sub("", hyp)
    if len(r) < k or not h:
        return False
    return r[-k:] in h[-window:]


def audit_tails(root: str):
    """复用 tail_audit 的口径，给同一批文件补尾音统计。"""
    db_all, ms_all, cut = [], [], 0
    for p in scan_audio(root):
        try:
            y, sr = sf.read(p, always_2d=True, dtype="float32")
            mono = y.mean(axis=1)
            tms, edb = end_metrics(mono, sr)
        except Exception:                                    # noqa: BLE001
            continue
        if edb is None:
            continue
        db_all.append(edb)
        ms_all.append(tms)
        if edb > -12.0:
            cut += 1
    n = len(db_all)
    if n == 0:
        return None
    srt, mss = sorted(db_all), sorted(ms_all)
    return {"n": n, "median_end_db": round(srt[n // 2], 1),
            "median_tail_ms": round(mss[len(mss) // 2], 1),
            "cut_ratio": round(cut / float(n), 3)}


def eval_side(root: str, refs: dict, model_obj, lang: str, k: int, limit: int):
    files = scan_audio(root)
    if not files:
        return {"ok": False, "error": f"目录里没有 wav/flac: {root}"}
    if limit > 0:
        files = files[:limit]

    rows, cers, hits = [], [], 0
    hit_n = 0
    for p in files:
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        ref = refs.get(name) or refs.get(stem) or ""
        try:
            segs, _info = model_obj.transcribe(p, language=None if lang == "auto" else lang,
                                               beam_size=5)
            text = "".join(s.text for s in segs).strip()
        except Exception as e:                               # noqa: BLE001
            rows.append({"file": name, "error": f"{type(e).__name__}: {e}", "ref": ref})
            continue
        c = cer(ref, text) if ref else None
        h = tail_hit(ref, text, k) if ref else None
        if c is not None:
            cers.append(c)
        if h is not None:
            hit_n += 1
            hits += int(h)
        rows.append({"file": name, "ref": ref, "asr": text,
                     "cer": c, "tail_hit": h})

    n = len(rows)
    bad = sorted((r for r in rows if r.get("cer") is not None), key=lambda x: -x["cer"])[:10]
    out = {
        "dir": os.path.abspath(root),
        "n": n,
        "with_ref": len(cers),
        "cer_mean": round(sum(cers) / len(cers), 3) if cers else None,
        "cer_median": round(sorted(cers)[len(cers) // 2], 3) if cers else None,
        "tail_hit_rate": round(hits / hit_n, 3) if hit_n else None,
        "tail_stats": audit_tails(root),
        "worst_cer": bad,
    }
    return {"ok": True, **out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="无参考验收：ASR 回转写 + CER + 末尾字覆盖")
    ap.add_argument("--dir", required=True, help="A 侧 wav 目录")
    ap.add_argument("--dir-b", default="", help="B 侧 wav 目录（可选，给了就出对比）")
    ap.add_argument("--refs", default="", help="参考文本 json：{文件名或词干: 台词}")
    ap.add_argument("--k", type=int, default=3, help="末尾字覆盖的字符数（默认 3）")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="", help="HF_HOME")
    ap.add_argument("--allow-download", action="store_true",
                    help="本地无模型缓存时允许联网下载（默认离线：安全默认，"
                         "不偷偷拉几个 GB 的权重）")
    ap.add_argument("--limit", type=int, default=0, help="每侧最多评多少条（0=全量）")
    ap.add_argument("--json", default="", help="报告写盘路径")
    args = ap.parse_args(argv)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # torch + whisper 常见冲突
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from faster_whisper import WhisperModel

    refs = load_refs(args.refs)
    model_obj = WhisperModel(args.model, device=args.device,
                             compute_type=args.compute_type)

    a = eval_side(args.dir, refs, model_obj, args.lang, args.k, args.limit)
    if not a.get("ok"):
        print(json.dumps(a, ensure_ascii=False))
        return 3

    out = {"ok": True, "model": args.model, "k": args.k, "a": a}
    if args.dir_b:
        b = eval_side(args.dir_b, refs, model_obj, args.lang, args.k, args.limit)
        if not b.get("ok"):
            print(json.dumps({**b, "a": a}, ensure_ascii=False))
            return 3
        out["b"] = b
        out["compare"] = {
            "cer_mean": [a.get("cer_mean"), b.get("cer_mean")],
            "tail_hit_rate": [a.get("tail_hit_rate"), b.get("tail_hit_rate")],
            "median_end_db": [a.get("tail_stats", {}).get("median_end_db"),
                              b.get("tail_stats", {}).get("median_end_db")],
            "cut_ratio": [a.get("tail_stats", {}).get("cut_ratio"),
                          b.get("tail_stats", {}).get("cut_ratio")],
        }

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        out["report"] = args.json
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
