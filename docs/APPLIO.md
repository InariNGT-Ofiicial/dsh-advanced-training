# Applio 训练路线（0.4.0）

`adv_train route=applio` 直接运行本地 Applio 的原生训练脚本，完成切片、F0/特征、训练、续训、索引和模型交接。游戏解包、番剧摘句后的干净单角色音频，以及自己的语音集均可接入。已有外部实验通过 adopt 登记后直接 resume，不需要打开 WebUI 点开始。

## 先配置工程

本版核验 Applio 3.7.1，接口提交为 [fb427aa388a954a387da7034fbbc08313f7ffe8c](https://github.com/IAHispano/Applio/tree/fb427aa388a954a387da7034fbbc08313f7ffe8c)。8 个关键接口文件按换行归一后的 SHA-256 校验；其他版本或改动不匹配时 doctor 会报告并拒绝执行，不能直接套用位置参数。引擎、Python 环境和权重由使用者准备，不随插件分发。

在配置 JSON 的 `pipelines.applio` 填写以下片段，路径按本机修改：

```json
{
  "repo": "E:/Applio",
  "python": "E:/Applio/env/python.exe",
  "logsDir": "E:/Applio/logs",
  "workRoot": "D:/AudioWork",
  "modelDir": "D:/AudioWork/voice-models",
  "ffmpegDir": "D:/Tools/ffmpeg/bin",
  "embedderDir": "E:/Applio/rvc/models/embedders/contentvec",
  "pretrainedG": "E:/Applio/rvc/models/pretraineds/hifi-gan/f0G48k.pth",
  "pretrainedD": "E:/Applio/rvc/models/pretraineds/hifi-gan/f0D48k.pth",
  "expName": "my-applio-voice",
  "sampleRate": "48k",
  "vocoder": "HiFi-GAN",
  "precision": "fp16",
  "f0Method": "rmvpe",
  "cutMode": "Automatic",
  "normalization": "none",
  "epochs": 150,
  "batchSize": 4,
  "saveEvery": 10,
  "processes": 2,
  "minSlices": 1
}
```

支持配置 HiFi-GAN 的 32k/40k/48k、RefineGAN 的 24k/32k，必须备齐对应配置和底模。实际完整 GPU 验证为 Windows、HiFi-GAN、48k、RMVPE、FP16、单 GPU 0；其他组合是已接入的参数范围，尚未逐一完成训练验收。切片可用 Automatic 或 Skip，归一化可用 none/pre/post；Simple 切片未暴露。minSlices 仅用于拒绝空产物，不表示少量语料足够训练。

训练环境需 Applio 自身依赖、本地 embedder 配置与权重、所选 F0 模型和对应静音项。默认不在线下载，显式 embedder 路径错误不会换成其他模型。每个后台任务使用独立的 Applio 设置文件，并固定 APPLIO_LOGS_DIR，避免读取 WebUI 的另一份目录或清理选项。环境变量也可用 DSH_APPLIO_ROOT / DSH_APPLIO_PYTHON。

`logsDir` 填实验根目录，不是某一个实验文件夹。若 Applio 迁移过模型目录，必须填实际位置；doctor 会显示本次使用的路径。阶段参数不切换采样率，修改采样率时同时调整底模等路线配置。

## 从新语音集开始

先用 `adv_train_catalog route=applio` 查看真实参数。以下 JSON 均为 `adv_train` 的 arguments：

```json
{"route":"applio","action":"doctor","params":{"deep":true}}
```

```json
{"route":"applio","action":"scan","params":{"input":"D:/AudioWork/角色A"}}
```

回听确认角色、清晰度、配乐/重叠人声和句尾，再执行：

```json
{"route":"applio","action":"build","params":{"input":"D:/AudioWork/角色A","expName":"role-a"}}
```

build 返回 jobId，查询同一实验的 status，等 succeeded 后才进入下一阶段。preprocess、run、index 同样是后台操作，都必须查询各自 jobId。

| action | 完成内容与检查 |
| --- | --- |
| build | 调用原生预处理，验证 sliced_audios 的数量、单声道与采样率；保留源音频 |
| preprocess | 调用原生 F0/特征提取，验证 extracted 的 768 维特征、f0/f0_voiced、配置、单角色清单和静音项 |
| run | 新训练，固定 GPU 0、cleanup=false；已有权重拒绝重开；核验目标轮次的导出模型及成对 G/D |
| index | 调用原生索引构建并读取 `<expName>.index`，检查 768 维且有向量 |
| register | 交接模型、索引、配置、清单、模型信息、训练日志、恢复证明与哈希，不覆盖既有音色包 |

例如开始新训练：

```json
{"route":"applio","action":"run","params":{"expName":"role-a","epochs":150,"batchSize":4,"saveEvery":10}}
```

```json
{"route":"applio","action":"status","params":{"expName":"role-a","jobId":"<本次返回的 jobId>"}}
```

本入口不暴露 cleanup、自动关机、推理和翻唱。原生训练会正常写入该实验的检查点和配置，因此请使用独立实验名，不与外部运行任务共用。

索引算法仍为原生实现；Windows 中文路径的保存使用 FAISS VectorIOWriter 转字节后由 Python 原子写盘，避开上游窄字符文件接口。中文 logsDir 已在完整流程中实测。

## 接管已有实验并继续训练

先停止原来的外部训练，再通过插件查看和接管：

```json
{"route":"applio","action":"experiments","params":{}}
```

```json
{"route":"applio","action":"adopt","params":{"expName":"role-a"}}
```

adopt 检查单角色音频/特征/配置、G/D 配对、真实轮次与优化器状态，记录到插件工作区，不启动训练，也不清理权重。检查到存活的外部训练进程时拒绝接管。列表只展示 G/D 是否存在，不能以文件名推断实际恢复轮次。

只继续已有实验时，doctor 可传 `resumeOnly=true`，无需重新准备提取模型或新训练底模；adopt 仍会核对该实验清单实际引用的音频和特征。

```json
{"route":"applio","action":"resume","params":{"expName":"role-a","epochs":200}}
```

resume 自动执行：

1. 获取本实验锁，核对实际 logsDir 和工作记录。
2. 将最新成对 G/D、配置、清单、model_info 及已有 training_data 备份到 `<workRoot>/applio/<expName>/checkpoint-backups/<UTC时间>-<jobId>/`，该位置必须在实际 logsDir 之外。
3. 核验源文件稳定性、大小、SHA-256、成对轮次和优化器状态。空间不足、缺失或不一致即停止。
4. 从快照严格恢复模型、优化器和 AMP 状态；兼容 Applio 的五项返回值与参数化权重命名。恢复异常不进入从第 1 轮重开的兜底，120 秒内未收到确认也会停止。
5. 核对恢复证明、目标轮次模型和 G/D。status 显示备份位置、fromEpoch、targetEpoch 与 confirmed；任务目录保留 resume-proof.json。

epochs 是目标总轮次，须大于已保存轮次。备份独立保留，不自动覆盖或清理，需要预留完整 G/D 对应空间。`run resume=true` 也走同一保护；直接使用 resume 更清楚，不需要去外部界面点击开始。

也可用 `adv_train_plan` 的 `route=applio, sourceType=experiment, source=role-a` 生成接管/续训计划；计划本身不执行。

## 状态、停止和交接

stop 仅停止本入口启动的任务，传同一 expName/jobId，并继续用 status 确认 stopped。Windows 使用独立 Job Object，协调进程异常退出时结束其拥有的子进程树；不按进程名结束外部任务。

模型交接使用 register，可传角色显示名 role 和新的 outDir。默认要求索引，明确只交接无检索模型时可设置 includeIndex=false。`.applio-manifest.json` 记录接口依据、模型和源配置，固定保留 qualityAccepted=false。

本轮只有合成语料的流程验证，不能等同于真实角色的音质、相似度或泛化验收。详见 [VALIDATION.md](VALIDATION.md)。
