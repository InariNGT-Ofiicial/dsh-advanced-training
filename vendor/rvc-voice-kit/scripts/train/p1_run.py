# -*- coding: utf-8 -*-
"""
RVC P1 贯通脚本：数据集 -> 六段 -> 可听模型
每段都校验产物，支持 --from 续跑。cwd 固定为 repo 根（train.py 里 ./logs 是硬编码）。
用法：
  python p1_run.py --steps prep          # 预处理
  python p1_run.py --steps prep,f0,feat,list   # 到 filelist
  python p1_run.py --from train --epochs 50    # 训练起
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402
from rvcx.paths import child_env as _child_env  # noqa: E402

# 沿用原脚本的短名，下面所有命令拼装保持不变
ROOT, REPO, PY = _paths.ROOT, _paths.REPO, _paths.PY

DATA = _paths.DATASET / "fangno"        # 可用 --data 覆盖
EXP = "fangno_p1"                      # ★ -e 只能传名字，不能传路径
LOGDIR = REPO / "logs" / EXP

ALL_STEPS = ["prep", "f0", "feat", "list", "train", "index", "infer"]


def sh(cmd, tag, timeout=None):
    """跑一条命令，实时落日志。"""
    log = ROOT / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{tag}] {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== {time.strftime('%F %T')} =====\n")
        fh.write(" ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        p = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(REPO),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        tail = []
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            line = line.rstrip()
            if line:
                tail.append(line)
                tail = tail[-25:]
                print("  " + line[:200], flush=True)
        p.wait(timeout=timeout)
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"=== [{tag}] {'OK' if ok else 'FAIL rc=%s' % p.returncode} | {dt:.1f}s", flush=True)
    if not ok:
        print("\n".join(tail), file=sys.stderr)
    return ok, dt


def count(d, pat="*"):
    p = Path(d)
    return len(list(p.glob(pat))) if p.is_dir() else 0


def step_prep(a):
    if count(LOGDIR / "1_16k_wavs", "*.wav") > a.min_slices:
        print(f"[prep] 已存在 {count(LOGDIR / '1_16k_wavs', '*.wav')} 个切片，跳过")
        return True
    sr = 48000 if a.sr == "48k" else 40000
    # ★ logs/<exp>/ 必须预先存在，preprocess.py 直接 open() 日志文件，不会自建
    LOGDIR.mkdir(parents=True, exist_ok=True)
    # ★ 第 5 位 "True" = 不并行，要并行必须传 False
    ok = sh([PY, "-m", "train.preprocess", str(DATA), sr, a.np, f"logs/{EXP}", "False", "3.7"], "01_prep")[0]
    n = count(LOGDIR / "1_16k_wavs", "*.wav")
    print(f"[prep] 切片产出 {n} 个")
    # ★ preprocess.py 内部吞异常、退出码仍为 0，只能靠数产物判断
    if n < a.min_slices:
        print(f"[prep] 产出过少（{n} < {a.min_slices}），按失败处理", file=sys.stderr)
        return False
    return ok


def step_f0(a):
    if count(LOGDIR / "2a_f0", "*.npy") > 50:
        print(f"[f0] 已存在 {count(LOGDIR / '2a_f0', '*.npy')} 个，跳过")
        return True
    # ★ 必须以 cpu/cuda 开头：cpu <exp_dir> <n_p> <method> / cuda <n_part> <i_part> <i_gpu> <exp_dir> <is_half>
    if a.f0method == "rmvpe":
        return sh([PY, "-m", "train.dataset.extract_f0", "cuda", "1", "0", "0", f"logs/{EXP}", "True"], "02_f0")[0]
    # pm 走 parselmouth，不依赖 rmvpe.pt（P1 用来先跑通链路）
    return sh([PY, "-m", "train.dataset.extract_f0", "cpu", f"logs/{EXP}", a.np, a.f0method], "02_f0")[0]


def step_feat(a):
    fd = LOGDIR / ("3_feature768" if a.version == "v2" else "3_feature256")
    if count(fd, "*.npy") > 50:
        print(f"[feat] 已存在 {count(fd, '*.npy')} 个，跳过")
        return True
    # device n_part i_part exp_dir version is_half
    return sh([PY, "-m", "train.dataset.extract_hubert_feature", "cuda:0", "1", "0",
               f"logs/{EXP}", a.version, "True"], "03_feat")[0]


def find_npy(d, stem):
    """兼容 xxx.wav.npy 与 xxx.npy 两种命名。"""
    for name in (f"{stem}.wav.npy", f"{stem}.npy"):
        p = d / name
        if p.exists():
            return p
    return None


def step_list(a):
    """★ 两个必须照 WebUI 来的点：① 绝对路径（data_utils 直接 getsize）② 注入 logs/mute 静音样本。"""
    gts = sorted((LOGDIR / "0_gt_wavs").glob("*.wav"))
    fea_dim = 768 if a.version == "v2" else 256
    feat_dir = LOGDIR / f"3_feature{fea_dim}"
    R = str(REPO).replace("\\", "/")
    E = f"{R}/logs/{EXP}"
    rows, miss = [], 0
    for g in gts:
        stem = g.stem
        f_feat = feat_dir / f"{stem}.npy"
        f_f0 = find_npy(LOGDIR / "2a_f0", stem)
        f_nsf = find_npy(LOGDIR / "2b-f0nsf", stem)
        if not (f_feat.exists() and f_f0 and f_nsf):
            miss += 1
            continue
        rows.append(f"{E}/0_gt_wavs/{g.name}|{E}/3_feature{fea_dim}/{stem}.npy|"
                    f"{E}/2a_f0/{f_f0.name}|{E}/2b-f0nsf/{f_nsf.name}|0")
    if not rows:
        raise SystemExit("[list] 没有任何完整样本，检查前面三步")

    mute = (f"{R}/logs/mute/0_gt_wavs/mute{a.sr}.wav|{R}/logs/mute/3_feature{fea_dim}/mute.npy|"
            f"{R}/logs/mute/2a_f0/mute.wav.npy|{R}/logs/mute/2b-f0nsf/mute.wav.npy|0")
    rows.extend([mute] * 2)   # WebUI 每说话人追加 2 条
    out = LOGDIR / "filelist.txt"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[list] 写入 {len(rows)} 行（含 2 条静音），缺料跳过 {miss} 条 -> {out}")
    return True


def step_train(a):
    small = list((REPO / "assets" / "weights").glob(f"{EXP}_e*.pth"))
    if small and not a.resume:
        print(f"[train] 已有小模型 {small[0].name}，跳过")
        return True
    # ★ train.py 读 logs/<exp>/config.json，WebUI 训练前会从 configs/ 复制，CLI 必须自己补
    cfg_src = REPO / "configs" / ("v1" if (a.version == "v1" or a.sr == "40k") else "v2") / f"{a.sr}.json"
    cfg_dst = LOGDIR / "config.json"
    if not cfg_dst.exists():
        if not cfg_src.exists():
            raise SystemExit(f"[train] 缺超参模板 {cfg_src}")
        shutil.copy(cfg_src, cfg_dst)
        print(f"[train] 复制超参模板 {cfg_src.name} -> {cfg_dst}")
    gen = REPO / "assets" / "pretrained_v2" / ("f0G48k.pth" if a.sr == "48k" else "f0G40k.pth")
    dis = REPO / "assets" / "pretrained_v2" / ("f0D48k.pth" if a.sr == "48k" else "f0D40k.pth")
    for p in (gen, dis):
        if not p.exists():
            raise SystemExit(f"[train] 缺底模 {p}")
    return sh([PY, "-m", "train.train",
               "-e", EXP, "-sr", a.sr, "-f0", "1", "-bs", str(a.bs), "-g", "0",
               "-te", str(a.epochs), "-se", str(a.se),
               "-pg", f"assets/pretrained_v2/{gen.name}",
               "-pd", f"assets/pretrained_v2/{dis.name}",
               "-l", "1", "-c", "0", "-sw", "1", "-v", a.version], "04_train")[0]


def step_index(a):
    if list(LOGDIR.glob("added_*.index")):
        print("[index] added 索引已存在，跳过")
        return True
    idx_root = REPO / "assets" / "indices"
    idx_root.mkdir(parents=True, exist_ok=True)
    # 位置参数：exp_name version outside_index_root n_cpu
    return sh([PY, "-m", "train.train_index", EXP, a.version, str(idx_root).replace("\\", "/"), "8"], "05_index")[0]


def step_infer(a):
    w = REPO / "assets" / "weights"
    small = sorted(w.glob(f"{EXP}_e*.pth"))
    if not small:
        raise SystemExit("[infer] 没有小模型，训练没产出？")
    model = small[-1]
    idx = sorted(LOGDIR.glob("added_*.index"))
    if not idx:
        raise SystemExit("[infer] 没有 added 索引")
    src = ROOT / "probe" / "in"
    src.mkdir(parents=True, exist_ok=True)
    if not any(src.glob("*.wav")):
        raise SystemExit(f"[infer] 请先放一条干声到 {src}")
    out = ROOT / "probe" / "out"
    cmd = [PY, "-m", "infer.cli", "--model", str(model), "--index", str(idx[-1]),
           "--input", str(src), "--output", str(out), "--pitch", str(a.pitch),
           "--f0-method", "rmvpe", "--index-rate", str(a.index_rate),
           "--protect", "0.33", "--format", "wav", "--overwrite"]
    return sh(cmd, "06_infer")[0]


STEPS = {"prep": step_prep, "f0": step_f0, "feat": step_feat, "list": step_list,
         "train": step_train, "index": step_index, "infer": step_infer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(ALL_STEPS[:4]), help="逗号分隔，或 all")
    ap.add_argument("--from", dest="frm", default=None, help="从某步跑到末尾")
    ap.add_argument("--sr", default="48k", choices=["48k", "40k"])
    ap.add_argument("--version", default="v2", choices=["v2", "v1"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--se", type=int, default=10, help="每 N epoch 存一次")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--np", default="8", help="预处理进程数")
    # ★ 50 是生产阈值（24.8 min 训练集能切出 445 片）。做小规模探针时按比例调低，
    #   否则数据链明明通了也会被判失败——「产出过少」这个门是防静默失败，不是技术下限。
    ap.add_argument("--min-slices", type=int, default=50, help="切片数下限，低于此判失败")
    ap.add_argument("--f0method", default="pm", choices=["pm", "rmvpe"])
    ap.add_argument("--exp", default=None, help="实验名（logs/<exp>），不同轮次务必分开")
    ap.add_argument("--data", default=None, help="训练集目录，默认 <DATASET>/fangno")
    ap.add_argument("--resume", action="store_true", help="Explicitly resume an existing experiment")
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--index-rate", default="0.75")
    a = ap.parse_args()

    global EXP, LOGDIR, DATA
    if a.data:
        DATA = Path(a.data)
        print(f"[data] 训练集 -> {DATA}")
    if a.exp:
        EXP = a.exp
        LOGDIR = REPO / "logs" / EXP

    steps = ALL_STEPS[ALL_STEPS.index(a.frm):] if a.frm else (
        ALL_STEPS if a.steps == "all" else [s.strip() for s in a.steps.split(",") if s.strip()])
    print(f"exp={EXP} steps={steps} sr={a.sr} ver={a.version} epochs={a.epochs} bs={a.bs} f0={a.f0method}")
    for s in steps:
        if s not in STEPS:
            raise SystemExit(f"未知步骤 {s}")
        t0 = time.time()
        ok = STEPS[s](a)
        print(f"--> {s}: {'OK' if ok else 'FAIL'} ({time.time() - t0:.1f}s)", flush=True)
        if not ok:
            raise SystemExit(f"步骤 {s} 失败，已停止")


if __name__ == "__main__":
    main()
