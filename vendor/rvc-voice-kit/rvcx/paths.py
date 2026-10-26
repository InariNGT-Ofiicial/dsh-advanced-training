# -*- coding: utf-8 -*-
"""集中解析所有外部路径。

本仓库**不含**模型权重、训练数据、上游 RVC 源码和 venv。所有路径按优先级解析：

    环境变量  >  rvcx.local.toml（不入库）  >  相对 ROOT 的中性默认值

环境变量
--------
RVCX_ROOT        工作根目录（repo / venv / 实验产物都在它下面）
RVCX_REPO        上游 RVC clone 目录，默认 <ROOT>/repo
RVCX_PY          venv 里的 python.exe，默认 <ROOT>/venv/Scripts/python.exe
RVCX_FFMPEG_DIR  装了 ffmpeg.exe/ffprobe.exe 的目录（RVC 代码写死 cmd=["ffmpeg"]）
RVCX_MODELS      音色模型根目录（每个角色一个子目录）
RVCX_DATASET     训练数据集根目录
RVCX_REF         LTAS 参考集目录（通常是目标角色训练集本身）
RVCX_WORK        实验产物根目录（dialogue / probe / cache），默认 <ROOT>

用法
----
    from rvcx.paths import ROOT, REPO, PY, FFMPEG_DIR, MODELS

`python -m rvcx.paths` 打印当前解析结果，排查「为什么找不到模型」时先跑它。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def _load_local() -> dict:
    """读 rvcx.local.toml。搜索顺序：RVCX_LOCAL 指定 > 仓库根 > 当前目录。"""
    if tomllib is None:
        return {}
    cands = []
    if os.environ.get("RVCX_LOCAL"):
        cands.append(Path(os.environ["RVCX_LOCAL"]))
    cands.append(Path(__file__).resolve().parent.parent / "rvcx.local.toml")
    cands.append(Path.cwd() / "rvcx.local.toml")
    for p in cands:
        if p.is_file():
            try:
                with open(p, "rb") as fh:
                    return tomllib.load(fh)
            except Exception as e:  # 配置写坏了要说清楚，不能静默回落
                print(f"[rvcx.paths] 读取 {p} 失败：{e}", file=sys.stderr)
    return {}


_LOCAL = _load_local()


def _resolve(env: str, key: str, default):
    """优先级：环境变量 > toml > 默认值。空值一律视为「未设置」，返回 None。

    ★ 不能返回 Path("")——它等于 Path(".")，是 truthy，会让「可选路径」看起来已配置。
    """
    v = os.environ.get(env) or _LOCAL.get(key)
    if v:
        return Path(str(v)).expanduser()
    if default is None or str(default) == "":
        return None
    return Path(default)


# ---- 工作根与上游 code ----------------------------------------------------
ROOT = _resolve("RVCX_ROOT", "root", Path(r"E:/AI-RVC"))
REPO = _resolve("RVCX_REPO", "repo", ROOT / "repo")
PY = _resolve("RVCX_PY", "python", ROOT / "venv" / "Scripts" / "python.exe")

# ffmpeg：为空表示「信任系统 PATH」。RVC 的 preprocess 写死 cmd=["ffmpeg"]，
# 只有把目录塞进子进程 PATH 才能生效，所以这里只需要目录，不需要可执行文件路径。
FFMPEG_DIR = _resolve("RVCX_FFMPEG_DIR", "ffmpeg_dir", "")

# ---- 资产 -----------------------------------------------------------------
MODELS = _resolve("RVCX_MODELS", "models", ROOT / "models")
DATASET = _resolve("RVCX_DATASET", "dataset", ROOT / "dataset")
REF = _resolve("RVCX_REF", "refdir", "")
WORK = _resolve("RVCX_WORK", "work", ROOT)

# ---- 可选：台词实验用的 TTS 引擎 -------------------------------------------
# 只有 scripts/dialogue/ 里的 TTS 脚本需要；不做台词实验可以留空。
COSY = _resolve("RVCX_COSY", "cosy_dir", "")            # CosyVoice2 安装目录
COSY_REF = _resolve("RVCX_COSY_REF", "cosy_ref", "")    # 参考音色 wav（zero-shot）
SOVITS = _resolve("RVCX_SOVITS", "sovits_dir", "")      # GPT-SoVITS 安装目录

# ---- 可选：DDSP-SVC 对拍（scripts/eval/compare_ddsp.py）-------------------
DDSP_ROOT = _resolve("RVCX_DDSP", "ddsp_root", "")

# ---- 上游版本（写进 README 与 setup 脚本，勿随意改）------------------------
UPSTREAM = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
UPSTREAM_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
UPSTREAM_TAG = "RVC 2.3.260718"


def child_env() -> dict:
    """子进程环境：REPO 进 PYTHONPATH（train/ 用绝对导入）+ ffmpeg 目录进 PATH。"""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    if FFMPEG_DIR and (Path(FFMPEG_DIR) / "ffmpeg.exe").exists():
        env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    return env


def require(*paths, what: str = "路径") -> None:
    """缺依赖时给一句能照着修的话，而不是让下游抛 KeyError/FileNotFoundError。"""
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(
                f"[rvcx] {what}不存在：{p}\n"
                f"         检查环境变量 RVCX_ROOT / RVCX_REPO / RVCX_MODELS，"
                f"或写一份 rvcx.local.toml（模板见 rvcx.local.example.toml）"
            )


def main() -> None:
    rows = [
        ("ROOT", ROOT, True),
        ("REPO", REPO, True),
        ("PY", PY, True),
        ("FFMPEG_DIR", FFMPEG_DIR, False),
        ("MODELS", MODELS, False),
        ("DATASET", DATASET, False),
        ("REF", REF, False),
        ("WORK", WORK, True),
    ]
    print(f"rvcx.paths  (local toml: {'已加载' if _LOCAL else '未找到'})\n")
    for name, val, must in rows:
        if val is None:
            print(f"  {name:12s} = <未设置>")
            continue
        ok = Path(val).exists()
        mark = "OK " if ok else ("MISS" if must else "--  ")
        print(f"  {name:12s} = {val}   [{mark}]")


if __name__ == "__main__":
    main()
