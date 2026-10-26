# dsh-advanced-training

DeepseekHarness 中面向音色模型训练的统一入口。一个主插件接入游戏素材、番剧摘句和自备单人语音集，衔接数据整理、质量检查、DDSP / GPT-SoVITS / RVC 训练与模型交接。

**[v0.3.0](https://github.com/InariNGT-Ofiicial/dsh-advanced-training/releases/tag/v0.3.0)** 新增 RVC v2 训练、续训、检索索引与音色包交接。原四个质量工具保持兼容，旧版说明存于 [质量方法与历史实验](docs/QUALITY_METHODS.md)。

## 生态位置

DeepseekHarness 对话 → 本插件统一组织/派发 → 对应素材与训练后端 → 模型、配置、日志交接。

| 内置来源 | 承担的工作 |
| --- | --- |
| [dsh-train-pipeline 0.5.0](https://github.com/InariNGT-Ofiicial/dsh-train-pipeline) | 兼容封包解包、角色盘点、DDSP 出料、质量检查、预处理、训练、状态与模型注册 |
| [dsh-anime-voice](https://github.com/InariNGT-Ofiicial/dsh-anime-voice) 摘句/整集重建脚本 | 候选说话人、人工角色映射、目标切片、台词清单与训练输入桥接 |
| 同仓库 `tts-train-pipeline` 0.2.1 | GPT-SoVITS 原文配对出料、无文本语音 ASR 回标、预处理、训练、状态与模型注册 |
| [rvc-voice-kit](https://github.com/InariNGT-Ofiicial/rvc-voice-kit) 训练脚本 | RVC 原生出料、F0 / HuBERT 特征、训练清单、GPU 训练与续训、检索索引；主插件补充任务管理与产物核验 |

相关项目的固定运行源码已内置，不必分别注册训练插件。具体算法和训练仍由 DDSP-SVC、GPT-SoVITS、RVC 等上游工程执行；不同路线的模型不合并或互转。来源提交和文件哈希见 [manifest](vendor/manifest.json)，许可见 [第三方说明](THIRD_PARTY_NOTICES.md)。

外部工程、Python 环境、解包工具与预训练权重需要自行配齐；加载插件不会安装依赖、下载模型、启动训练或生成音频。

## 安装与配置

```powershell
git clone https://github.com/InariNGT-Ofiicial/dsh-advanced-training.git
cd dsh-advanced-training
# 将 training.example.json 复制到自己的训练工作区，填写本机路径。
$env:DSH_ADVANCED_CONFIG = 'D:/AudioWork/training.local.json'
dsh plugin add .
```

重载 DeepseekHarness。`DSH_ADVANCED_CONFIG` 指向本地 JSON，也可在 Cordis 配置设置 `trainingConfig`；文件内容优先于 bundle patch 默认值。未指定文件时沿用既有配置/环境变量。

模板：[training.example.json](training.example.json)。

- 共用：`songRoot/workRoot/receiptDir/ffmpeg`。
- DDSP：`ddspDir/ddspEnv`；`pipelines.ddsp` 配独立 `trainRoot`、实验名、配置文件、模型库。
- GPT-SoVITS：`gsvDir/gsvEnv`；`pipelines.gsv` 配实际已准备的版本、实验名和训练参数。模板 v2 为示例，不代替权重。
- RVC：`pipelines.rvc` 配 `root/repo/python`、`workRoot/ffmpegDir/modelDir` 和训练参数。支持已有 RVC v2、48k/40k、单 GPU 0、pm/rmvpe；具体工程布局与步骤见 [RVC 使用说明](docs/RVC.md)。40k 必须另备对应底模、配置与静音样本，本轮实际验证的是 48k。
- 质量检查：`pyEnv` 需 numpy/soundfile，旧 ASR 评估用 `gsvEnv`。
- 番剧：`pipelines.anime` 配分析 Python、FFmpeg、本地 `whisperModel`，以及含 `campplus_cn_common.bin` 的 `camppModel`。整集重建另需 `sepRoot/sepPython/sepConfig/sepCheckpoint`。参见 [分析依赖](vendor/anime-voice/requirements-analysis.txt)。

本地路径、音频、权重和配置不提交 Git。统一入口发现显式配置的训练工程路径不存在时会报错，不静默换成另一份工程。

## 工具与执行

主插件注册 7 个工具：

| 工具 | 功能 |
| --- | --- |
| `adv_train_catalog` | 返回已接入路线、操作和真实参数 schema，可按 route/action 查询 |
| `adv_train_plan` | 从游戏、番剧、自备语音制定逐步流程，标出人工检查和需要替换的实际输出；只制定计划 |
| `adv_train` | 实际派发内置后端，校验 params，返回后端结果和持久回执 |
| `adv_doctor` | 保留原三线体检；只用一条训练线时优先调用该路线 doctor |
| `adv_tail_audit` | 保留句尾完整性检查，源切尾需回源重建 |
| `adv_unk_patch` | 保留 GPT-SoVITS 专用文本修正，可计划、应用与回滚 |
| `adv_ab_eval` | 保留对已有音频的评估接口，不合成音频，也不进入默认训练流程 |

对话示例：“用这个目录里角色 A 的语音训练 DDSP 音色，先检查环境和数据，再按报告推进，保留模型、配置与日志。”

工具调用示例：

```json
{"tool":"adv_train_plan","arguments":{"route":"ddsp","sourceType":"audio","source":"D:/AudioWork/voice","speaker":"角色A"}}
```

```json
{"tool":"adv_train_catalog","arguments":{"route":"ddsp","action":"build"}}
```

```json
{"tool":"adv_train","arguments":{"route":"ddsp","action":"scan","params":{"input":"D:/AudioWork/voice"}}}
```

```json
{"tool":"adv_train","arguments":{"route":"ddsp","action":"build","params":{"input":"D:/AudioWork/voice","speakers":["角色A"]}}}
```

`ddsp/gsv/rvc` 是训练路线，`anime` 准备素材。params 以 catalog 的后端 schema 为准，不把 DDSP 的 `input/speakers` 原样传给 GPT-SoVITS 的 `lines/voiceDir/speaker`。RVC 接收单角色音频目录，不需要台词标注。

`dryRun=true` 只校验参数和预览派发，不执行，不证明环境/产物通过。`run` 成功只代表后端接受/启动任务，需继续 `status` 查进程、日志和检查点。DDSP / GPT-SoVITS 的 `stop/register` 沿用后端保护。RVC 的 `build/preprocess/run/index` 都在后台执行，返回 `jobId`；每步等 `status` 为 `succeeded` 后再继续，`stop` 只处理本入口拥有的任务。

统一回执默认存于 `<workRoot>/advanced-training/receipts/`，包含实际后端、参数、起止时间和结果。后端训练日志仍由原管线保存。

## 三类素材

**自备单人语音**：DDSP 用 `scan → build → qc → preprocess → run → status → register`。GPT-SoVITS 有原台词时用 JSONL 配对出料；无台词时 `build voiceOnly=true allowEmptyText=true → asr`，人工校对和尾音检查后预处理。空文本 WAV 不能当作已配对语料。

RVC 用 `doctor → scan → 语料回听/尾音检查 → build → preprocess → run → index → register`，各后台阶段之间查询对应 `jobId` 的 `status`。默认不覆盖已有出料或重开已有模型；续训需显式 `resume=true`。归档包含导出的小模型、added 索引、配置、清单、训练日志和哈希，不把 G/D 检查点当作可交接的小模型。

**游戏素材**：先 `ddsp/unpack`，读取实际解包目录再扫描。GPT-SoVITS 优先用 `gsv/script` 保留游戏剧本原文。解包支持范围由上游实现决定，不承诺所有游戏或加密格式。

**番剧素材**：`anime/extract` 导出候选说话人和台词；人工确认映射后 `anime/refine` 导出目标切片。目标角色音频可接 DDSP / RVC；GPT-SoVITS 使用 `anime/bridge` 将相对 `file` 或标准 `train.list` 转成训练器识别的绝对 `voice_path`，逐条核对音频存在，保留原台词、时间和来源。

含配乐或重叠对白时，不因自动摘句就认定适合训练。`anime/rebuild` 复用整集分离/ASR/切片/声纹筛选/报告流程；输入 `workDir/src/<episode>.m4a`、显式片段范围和人工确认的参考音频目录。输出固定为 `<workDir>/_full/out/ts/train.list`，不执行原脚本训练/推理阶段；拒绝覆盖已有非空最终训练集目录。

## 使用边界

- 新统一入口止于素材接入、数据整理、质量检查、对应模型训练与交接，不派发推理、配音、翻唱或发布；旧评估接口只消费已有产物。
- 一站式指在同一主插件中组织和调用，不保证一次请求无人介入跑完。角色、台词与素材质量仍需回听和校对。
- 模型出现/注册成功只证明文件存在或可发现，不证明音质达标。不同模型格式不互通，不用早期日语样例替代本次训练验收。
- RVC 本版接入 v2 单角色训练编排；合成语料已经真实 GPU 训练与续训验证，尚未进行真实角色的长期训练、推理试听或主观音质验收。音色包固定标记 `qualityAccepted=false`。
- 不分发模型、环境或私人素材。番剧适配器要求已下载的本地 ASR/声纹模型。

## 验证

```powershell
npm test
# 可选：只用合成语料跑真实 Python/FFmpeg 扫描和出料，不启动训练。
$env:TEST_AUDIO_PYTHON = '<装有 numpy/soundfile 的 python>'
$env:TEST_FFMPEG = '<ffmpeg 可执行文件>'
npm run test:local
# 可选：会实际启动 RVC GPU 训练；需先填写已安装的工程和独立工作目录。
$env:TEST_RVC_ROOT = '<RVC 安装根>'
$env:TEST_RVC_PYTHON = '<RVC 环境 python.exe>'
$env:TEST_RVC_WORK = '<新的验证工作目录>'
$env:TEST_RVC_FFMPEG_DIR = '<ffmpeg 所在目录>'
npm run test:rvc:local
```

`npm test` 覆盖契约、真实后端注册/派发、参数拒绝、执行回执、路径保护、番剧参数/产物检查和台词桥接，以及 RVC 任务归属、停止请求、异常状态与配置合并。普通测试不启动训练；番剧模型执行和 RVC 进程执行使用 mock。实测与未验证部分见 [验证记录](docs/VALIDATION.md)。

## License

主插件仍为 UNLICENSED。内置来源保留各自声明，本次整合不改变许可，也不将代码许可扩展到模型或素材。
