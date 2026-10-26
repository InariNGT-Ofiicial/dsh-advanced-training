# dsh-advanced-training

「先进训练方法」独立 DSH 插件（DDSP / GPT-SoVITS / RVC 三线）。

把散落在训练管线里、**实测有效**的三条方法收拢成独立工具集，与出料/训练插件解耦。
所有方法都有同批 383 条语料的对照数据背书（2026-10-24 实验）：

| 方法 | 旧做法 | 新做法 | 效果 |
|---|---|---|---|
| 音频留尾 | 双向裁到「最后有声帧 + 50ms」，每条停在满电平 | 只裁句首、保留句尾自然衰减、兜 0.16s 静音底 | 末三字念出率 **33% → 93%**，PER **0.197 → 0.018** |
| 消 UNK | `「」『』～―（）()` 与 `#` 全部降级成 UNK，所有词边界共享同一 embedding | 剥无音值符号、过滤 `#`（不进符号表，避免底模维度不匹配） | UNK 17 → 0 |
| 无参考验收 | 凭耳朵 A/B，听不出结构性缺陷 | ASR 回转写 → CER + 末尾字覆盖 + 尾音统计 | 可量化、可对比 |

## 工具集

| 工具 | 适用线 | 作用 |
|---|---|---|
| `adv_doctor` | 三线 | 工程体检：DDSP 整合包与科学计算 python、GSV 工程与 venv、rvc-voice-kit |
| `adv_tail_audit` | **三线通用** | 对已有数据集目录做句尾体检（递归扫 wav/flac），输出 `tail_ms / end_db / cut_ratio` 与判读 |
| `adv_unk_patch` | **GSV 专用** | 消 UNK 文本补丁（apply / revert / 幂等 / 备份 / 符号表核对） |
| `adv_ab_eval` | **三线通用** | 验收 A/B：对已合成的 wav 目录跑 ASR 回转写，产出 CER、末尾字覆盖率、尾音统计 |

**为什么三线通用**：RVC / DDSP / GPT-SoVITS 的数据集与合成产物落盘都是 wav——
体检与度量只看波形和转写，与模型线无关。唯一线绑定的是 UNK 补丁：
文本符号链路只有 GPT-SoVITS 有；DDSP/RVC 是 SVC 不吃文本，不存在 UNK 问题。

**为什么不做合成/训练**：那是各自管线插件的职责，这里不重复造轮子。
本插件只做「体检 → 修正 → 验收」闭环里的体检与验收两环，外加一个文本修正补丁。

## 三线组合表

| 环节 | DDSP | GPT-SoVITS | RVC |
|---|---|---|---|
| 出料（含留尾） | `train_build`（0.5.0 默认留尾） | `tts_build` `tailPad=true` | 数据集用留尾版切片 |
| 出料后体检 | `adv_tail_audit` | `adv_tail_audit` | `adv_tail_audit` |
| 训练前文本修正 | —（不吃文本） | `adv_unk_patch` | —（不吃文本） |
| 训练 | `train_run` | `tts_run` | `rvc-voice-kit` 六段编排 |
| 合成 A/B 产物 | `train_infer` | `tts_infer` | `python -m rvcx.voice convert` |
| 验收 | `adv_ab_eval` | `adv_ab_eval` | `adv_ab_eval` |

## 判读口径（与 dsh-train-pipeline 0.5.0 完全一致）

- `end_db`：最后有声采样前 40ms 的 RMS，相对全句 90 分位电平。**真人自然收尾约 −24dB**
- `end_db > −12dB` = 句子停在满电平（被切尾）
- `cut_ratio > 0.3` = 判「句尾被切断：训出来的模型会吞末字」
- 短促句（「あっ！」）停在较高电平是**正常的**——只看整体比例，别逐条判废

## 验收 A/B 的立场

1. **CER 绝对值不可迷信**：ASR 是会犯错的第三方（实测「朝武芳乃」被听成「友竹よしの」、
   「有地さん」被听成「フリッツさん」）。看 A vs B 同批句子的**相对差**。
2. **末尾字覆盖（tail_hit）是结构性信号**：ref 去标点后取末 k 个字符（默认 3），
   在 ASR 去标点文本的末 8 字符窗口里找子串。切尾训出的模型会吞末字，这个指标直接抓它。
3. **尾音统计一并看**：A/B 产物本身的 `cut_ratio` 能告诉你「模型是否学会了在满电平处断句」。

## 依赖

- `adv_tail_audit`：numpy + soundfile（默认用 DDSP 整合包的 env，`DSH_TRAIN_PYTHON` 可覆盖）
- `adv_unk_patch`：纯标准库，任何 python 都能跑
- `adv_ab_eval`：faster-whisper（GSV venv 里现成有，`DSH_GSV_PYTHON` 可覆盖）；
  **默认离线**（`HF_HUB_OFFLINE=1`），模型缓存自动发现 `<gsvDir>/../hf_cache`；
  本地没有时传 `allowDownload=true` 才联网下载——不偷偷拉几个 GB 的权重

## 本机实测记录（2026-10-26）

- `adv_tail_audit` 对芳乃留尾数据集：358 条，收尾中位 **−23.4dB**（贴真人 −24dB），
  满电平比例 0.017 → ✅ 正常
- 同一工具对 GSV 线旧切尾数据集（tts/datasets/yoshino）：满电平比例 **0.642** → ❌ 应重出料
- `adv_unk_patch` 对真实 GSV 工程：报 `already-external`（工程已有手工等价修复，不重复插入）；
  对纯原版副本 apply → 幂等 → revert 全部通过，revert 后与原版逐字节一致
- `adv_ab_eval` 对原声数据集 8 条：CER 0.171 / 末 3 字命中 75%（错误集中在专有名词，符合预期）

## 安装与自测

```bash
dsh plugin add <本目录>          # cordis.patch.yml 随包加载
npm test                         # 4 个工具契约 + 3 个 py 脚本语法
```

## 已知边界

- RVC 线本机只有 rvc-voice-kit 工具链，`adv_ab_eval` 对 RVC 产物已可用（wav 进 wav 出），
  但**尚未用真实 RVC 换声产物跑过全量验收**——工具对 wav 一视同仁，理论无差别，待实测背书。
- `adv_ab_eval` 的 ASR 单卡逐条转写，383 条全量约几十分钟；先用 `limit` 抽样看信号再全量。

## License

UNLICENSED。参见各上游仓库的许可说明；本插件不授予再分发权。
