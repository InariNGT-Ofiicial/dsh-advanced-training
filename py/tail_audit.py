#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tail_audit.py —— 对已有数据集/音频目录做「句尾体检」（独立于出料流程，三条模型线通用）。

背景（实测 2026-10-24，同批 383 条语料）：
  旧出料把每条数据双向裁到「最后有声帧 + 50ms」，每条都停在满电平，
  模型学成「句子在满电平处直接断」→ 末三字念出率 33%、PER 0.197。
  改为「只裁句首、保留句尾自然衰减、兜 0.16s 静音底」后 → 93% / 0.018。

判据（与 dsh-train-pipeline 0.5.0 的 end_metrics 完全同口径）：
  tail_ms : 最后一个有声采样到文件末尾的静音时长
  end_db  : 最后有声采样前 40ms 的 RMS 相对全句 90 分位电平
  cut     : end_db > -12dB（句子停在满电平）
  真人自然收尾约 -24dB。

适用范围：RVC / DDSP / GPT-SoVITS 的数据集都是 wav（或 flac）目录，
体检只看波形，与模型线无关——同一把尺子量三条线。

用法：
  python tail_audit.py --dir <wav 目录>            # 汇总 + 最差 20 条
  python tail_audit.py --dir ... --json report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print(json.dumps({"ok": False, "error": "缺少 soundfile（pip install soundfile）"},
                     ensure_ascii=False))
    sys.exit(2)

AUDIO_EXT = {".wav", ".flac"}
SKIP_EXT = {".bak_dsh_unk"}          # 防止把备份文件当数据扫进来


def end_metrics(mono, sr, win_sec: float = 0.04, rel: float = 0.01):
    """
    句尾体检，口径与 dsh-train-pipeline 0.5.0 的 end_metrics 完全一致：
    返回 (tail_ms, end_db)。end_db > -12dB = 句子停在满电平（被切尾）。

    单条高电平收尾不一定是错（「あっ！」这类短促句本来就这样），
    真正有意义的是**整体比例** cut_ratio，所以只进统计、不逐条判废。
    """
    if mono.size == 0:
        return None, None
    a = np.abs(mono)
    peak = float(a.max())
    if peak <= 0:
        return None, None
    idx = np.where(a > peak * rel)[0]
    if idx.size == 0:
        return None, None
    last = int(idx[-1])
    tail_ms = round((len(mono) - 1 - last) / float(sr) * 1000.0, 1)
    k = max(1, int(win_sec * sr))
    seg = a[max(0, last - k + 1): last + 1]
    rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
    ref = float(np.percentile(a, 90)) + 1e-9
    return tail_ms, round(20 * np.log10(rms / ref), 1)


def scan_audio(root: str):
    """递归收集音频文件；跳过隐藏目录与备份文件。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_")) or d == "_"]
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in AUDIO_EXT:
                continue
            if fn.endswith(".bak_dsh_unk"):
                continue
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


def audit_dir(root: str, limit_worst: int = 20):
    files = scan_audio(root)
    if not files:
        return {"ok": False, "error": f"目录里没有 wav/flac: {root}"}

    tail_db_all, tail_ms_all, cut_names = [], [], []
    per_file = []
    for p in files:
        rel = os.path.relpath(p, root)
        try:
            y, sr = sf.read(p, always_2d=True, dtype="float32")
            mono = y.mean(axis=1)
            tms, edb = end_metrics(mono, sr)
        except Exception as e:                                   # noqa: BLE001
            per_file.append({"file": rel, "error": f"{type(e).__name__}: {e}"})
            continue
        if edb is None:
            continue
        tail_db_all.append(edb)
        tail_ms_all.append(tms)
        per_file.append({"file": rel, "tail_ms": tms, "end_db": edb,
                         "cut": bool(edb > -12.0)})
        if edb > -12.0:
            cut_names.append(f"{rel}: 收尾 {edb}dB / 尾静音 {tms}ms")

    n = len(tail_db_all)
    if n == 0:
        return {"ok": False, "error": "所有文件都读不出尾音指标", "total": len(files)}

    srt = sorted(tail_db_all)
    ms = sorted(tail_ms_all)
    cut = len(cut_names)
    worst = sorted((x for x in per_file if x.get("cut")),
                   key=lambda x: -x["end_db"])[:limit_worst]
    return {
        "ok": True,
        "dir": os.path.abspath(root),
        "total": len(files),
        "measured": n,
        "read_errors": len([x for x in per_file if x.get("error")]),
        "tail_stats": {
            "n": n,
            "median_end_db": round(srt[n // 2], 1),
            "median_tail_ms": round(ms[len(ms) // 2], 1),
            "cut_ratio": round(cut / float(n), 3),
            "verdict": ("句尾被切断：训出来的模型会吞末字" if cut / float(n) > 0.3
                        else "正常（句尾有自然衰减）"),
        },
        "worst": worst,
        "cut_samples": cut_names[:20],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据集句尾体检（三线通用）")
    ap.add_argument("--dir", required=True, help="音频目录（递归扫描 wav/flac）")
    ap.add_argument("--json", default="", help="报告写盘路径")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(json.dumps({"ok": False, "error": f"目录不存在: {args.dir}"},
                         ensure_ascii=False))
        return 2

    out = audit_dir(args.dir)
    if args.json and out.get("ok"):
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        out["report"] = args.json
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
