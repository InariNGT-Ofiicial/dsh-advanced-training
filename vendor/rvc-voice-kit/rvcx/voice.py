# -*- coding: utf-8 -*-
"""rvcx.voice —— 一条命令跑完「干声 → 目标音色」的换声器。

本模块不实现模型，只把上游 RVC 的 `infer.cli` 包成一个不会踩坑的入口：
自动定位 repo、自动挂 ffmpeg、自动从模型目录的 `.rvc-manifest.json` 读推荐参数、
单文件/整目录统一处理、已完成的跳过。

    python -m rvcx.voice list
    python -m rvcx.voice doctor
    python -m rvcx.voice convert --voice 芳乃_RVC_步数120 --input dry/ --output wet/

★ 三条来自实测的硬约束（都已在代码里强制）：
  1. 子进程 cwd 必须是 repo 根，否则 train/infer 里的相对导入与 logs/ 路径全错。
  2. `infer.cli` 的 --input 是单文件时，--output 也必须是**文件路径**，
     传目录会以 "Unsupported output suffix" 失败。
  3. RVC 的 preprocess 写死 cmd=["ffmpeg"]，全靠子进程 PATH；ffmpeg 不在 PATH
     时报错信息与「输入文件坏了」很像，容易误判。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import paths

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}


# --------------------------------------------------------------------------- #
# 模型解析
# --------------------------------------------------------------------------- #
def _voice_dir(voice: str) -> Path:
    """`--voice` 既接受完整路径，也接受 MODELS 下的子目录名。"""
    p = Path(voice).expanduser()
    if p.is_dir():
        return p.resolve()
    q = Path(paths.MODELS) / voice
    if q.is_dir():
        return q.resolve()
    raise SystemExit(
        f"[voice] 找不到音色目录：{voice}\n"
        f"        已试过 {p} 和 {q}\n"
        f"        可用列表：python -m rvcx.voice list"
    )


def load_voice(voice: str) -> dict:
    """返回 {dir, model, index, manifest, rec}。索引缺失时 index=None（ir 必须为 0）。"""
    d = _voice_dir(voice)
    man = {}
    mf = d / ".rvc-manifest.json"
    if mf.is_file():
        man = json.loads(mf.read_text(encoding="utf-8"))

    model = d / "model.pth"
    if not model.is_file():
        # 兼容直接丢 G_*.pth 的情况；训练中间产物不是推理模型，但仍可用
        cands = sorted(d.glob("*.pth"))
        if not cands:
            raise SystemExit(f"[voice] {d} 里没有 model.pth")
        model = cands[0]

    idxs = sorted(d.glob("added_*.index"))
    index = idxs[0] if idxs else None
    if index is None and sorted(d.glob("trained_*.index")):
        print(f"[voice] 警告：{d.name} 只有 trained_*.index（训练中间产物），"
              f"未找到 added_*.index；index_rate 会被强制设为 0", file=sys.stderr)

    rec = dict(man.get("recommended_infer") or {})
    return dict(dir=d, model=model, index=index, manifest=man, rec=rec)


# --------------------------------------------------------------------------- #
# 任务编排
# --------------------------------------------------------------------------- #
def _collect(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    if not src.is_dir():
        raise SystemExit(f"[input] 不存在：{src}")
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"[input] {src} 下没有音频文件（支持 {', '.join(sorted(AUDIO_EXT))}）")
    return files


def _target(src: Path, root: Path, dst: Path, suffix: str) -> Path:
    """单文件 -> 输出路径；目录 -> 保持相对层级，写到目录里。"""
    if not root.is_dir():
        return dst if dst.suffix.lower() in AUDIO_EXT else dst / (src.stem + suffix + ".wav")
    rel = src.relative_to(root)
    return dst / rel.parent / (rel.stem + suffix + ".wav")


def convert_one(v: dict, src: Path, dst: Path, *, index_rate: float, pitch: int,
                f0: str, protect: float, dry_run: bool = False) -> bool:
    if dst.exists() and dst.stat().st_size > 1000 and not dry_run:
        print(f"  跳过（已存在）{dst.name}", flush=True)
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(paths.PY), "-m", "infer.cli", "--model", str(v["model"])]
    if v["index"] is not None and index_rate > 0:
        cmd += ["--index", str(v["index"])]
    cmd += ["--input", str(src), "--output", str(dst),
            "--pitch", str(pitch), "--f0-method", f0,
            "--index-rate", str(index_rate), "--protect", str(protect),
            "--format", "wav", "--overwrite"]

    if dry_run:
        print("  " + " ".join(cmd))
        return True

    r = subprocess.run(cmd, cwd=str(paths.REPO), env=paths.child_env(),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000
    if not ok:
        print(f"  !! 失败 {src.name} rc={r.returncode}", flush=True)
        print(((r.stdout or "") + (r.stderr or ""))[-800:], flush=True)
    return ok


def cmd_convert(a: argparse.Namespace) -> int:
    v = load_voice(a.voice)
    rec = v["rec"]

    # 参数优先级：显式传参 > manifest 推荐 > 内置默认
    f0 = a.f0 or rec.get("f0_method", "rmvpe")
    protect = a.protect if a.protect is not None else float(rec.get("protect", 0.33))
    pitch = a.pitch if a.pitch is not None else int(rec.get("pitch", 0))
    # ★ 内置默认 index_rate = 0（不是 0.75）。
    #   理由：added_IVF1929 检索在高音帧会匹配到**低八度**的相似向量，RVC 照着错音高合成
    #   （实测同一 15 s 窗口：ir=0 → 769.7 Hz 正确；ir=0.75 → 410.5 Hz 低八度）。
    #   且 ir=0 的音色保真**不降反升**（LTAS 相关 0.7356 vs 0.7283）。
    #   ⚠ 若某个 manifest 的 recommended_infer.index_rate 仍写着 0.75，这里会照读它 ⇒
    #     **改 manifest 时必须同步改这个字段**，否则缺陷会静默复活。
    ir = a.index_rate if a.index_rate is not None else float(rec.get("index_rate", 0.0))
    if v["index"] is None and ir > 0:
        print("[voice] 无索引，index_rate 归零", file=sys.stderr)
        ir = 0.0

    src = Path(a.input).expanduser()
    dst = Path(a.output).expanduser()
    files = _collect(src)

    print(f"音色   {v['dir'].name}")
    print(f"模型   {v['model'].name}  ({v['model'].stat().st_size / 1048576:.0f} MB)")
    print(f"索引   {v['index'].name if v['index'] else '<无>'}")
    print(f"参数   f0={f0} index_rate={ir} pitch={pitch} protect={protect}")
    print(f"任务   {len(files)} 个文件 → {dst}", flush=True)

    suffix = a.suffix
    ok = 0
    for s in files:
        ok += bool(convert_one(v, s, _target(s, src, dst, suffix),
                               index_rate=ir, pitch=pitch, f0=f0,
                               protect=protect, dry_run=a.dry_run))
    print(f"\n完成 {ok}/{len(files)}")
    return 0 if ok == len(files) else 1


def cmd_list(_: argparse.Namespace) -> int:
    root = Path(paths.MODELS)
    if not root.is_dir():
        print(f"模型根目录不存在：{root}\n用 RVCX_MODELS 指定", file=sys.stderr)
        return 1
    print(f"{root}\n")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = d / "model.pth"
        idx = sorted(d.glob("added_*.index"))
        if not m.is_file():
            continue
        man = {}
        if (d / ".rvc-manifest.json").is_file():
            try:
                man = json.loads((d / ".rvc-manifest.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        ep = (man.get("model") or {}).get("epochs", "?")
        print(f"  {d.name:28s} {m.stat().st_size / 1048576:5.0f} MB  "
              f"epochs={ep:<4} index={'有' if idx else '无'}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    """逐项体检，输出能照着修的结论。"""
    problems = 0

    def chk(name, ok, detail, fix=""):
        nonlocal problems
        print(f"  [{'OK' if ok else 'XX'}] {name:16s} {detail}")
        if not ok:
            problems += 1
            if fix:
                print(f"       → {fix}")

    print("环境体检\n")
    chk("python", True, sys.executable)
    chk("RVCX_ROOT", Path(paths.ROOT).is_dir(), str(paths.ROOT),
        "设 RVCX_ROOT 或写 rvcx.local.toml")
    chk("RVCX_REPO", (Path(paths.REPO) / "infer" / "cli.py").is_file(), str(paths.REPO),
        "git clone 上游 RVC 到该目录（见 setup.ps1）")
    chk("venv python", Path(paths.PY).is_file(), str(paths.PY), "python -m venv venv")
    chk("rmvpe 权重", (Path(paths.REPO) / "assets" / "rmvpe" / "rmvpe.pt").is_file(),
        "assets/rmvpe/rmvpe.pt", "上游仓库首次启动会自动下载，或手动放到该路径")

    ff = None
    if paths.FFMPEG_DIR:
        c = Path(paths.FFMPEG_DIR) / "ffmpeg.exe"
        if c.is_file():
            ff = str(c)
    if ff is None:
        import shutil
        ff = shutil.which("ffmpeg")
    chk("ffmpeg", bool(ff), ff or "未找到",
        "设 RVCX_FFMPEG_DIR 指向含 ffmpeg.exe 的目录；RVC 代码写死 cmd=['ffmpeg']")

    models = Path(paths.MODELS)
    chk("模型根目录", models.is_dir(), str(models), "设 RVCX_MODELS")
    if models.is_dir():
        n = len(list(models.glob("*/model.pth")))
        chk("可用音色", n > 0, f"{n} 个", "把 model.pth + added_*.index 放成 <MODELS>/<角色>/")

    print(f"\n{'通过' if problems == 0 else f'{problems} 项待修'}")
    return 0 if problems == 0 else 1


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rvcx.voice",
                                 description="RVC 换声器：干声 → 目标音色")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="执行换声")
    c.add_argument("--voice", required=True, help="音色目录（路径或 MODELS 下的名字）")
    c.add_argument("--input", required=True, help="干声文件或目录")
    c.add_argument("--output", required=True, help="输出文件或目录")
    c.add_argument("--index-rate", type=float, default=None, help="默认取 manifest 的 0.75")
    c.add_argument("--pitch", type=int, default=None, help="半音；0=保持原调（台词推荐）")
    c.add_argument("--f0", default=None, choices=["rmvpe", "pm", "harvest", "crepe"],
                   help="默认 rmvpe")
    c.add_argument("--protect", type=float, default=None, help="辅音保护 0–0.5，默认 0.33")
    c.add_argument("--suffix", default="", help="输出文件名追加后缀，如 _rvc")
    c.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    c.set_defaults(func=cmd_convert)

    sub.add_parser("list", help="列出可用音色").set_defaults(func=cmd_list)
    sub.add_parser("doctor", help="环境体检").set_defaults(func=cmd_doctor)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
