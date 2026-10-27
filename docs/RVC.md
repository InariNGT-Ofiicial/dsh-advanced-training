# RVC 路线（0.3.1）

主插件调用固定版本 rvc-voice-kit 的训练脚本，训练计算由本机 RVC 引擎完成。RVC 接收已确认的单角色音频，不需要文本；游戏解包和番剧摘句得到的干净角色目录、自备语音目录都可以作为输入。角色混杂、配乐、人声重叠或切尾仍需回源整理和回听。

本版支持 v2、F0、48k / 40k、单 GPU 0，F0 方法为 pm 或 rmvpe。本机实测为 Windows、RVC `train.*` 模块布局、48k、pm、RTX 5060 Laptop（8GB）、PyTorch 2.7.1+cu128。40k、rmvpe 和其他系统未在本轮做完整训练验收。RVC 工程各分支并不统一，doctor 通过仅代表检查项齐备，后续阶段仍以产物核验为准。

## 配齐依赖

在 `training.example.json` 的 `pipelines.rvc` 中填写：

```json
{
  "root": "E:/AI-RVC",
  "repo": "E:/AI-RVC/repo",
  "python": "E:/AI-RVC/venv/Scripts/python.exe",
  "workRoot": "D:/AudioWork",
  "ffmpegDir": "D:/Tools/ffmpeg/bin",
  "modelDir": "D:/AudioWork/voice-models",
  "expName": "my-rvc-voice",
  "sampleRate": "48k",
  "f0Method": "rmvpe",
  "epochs": 150,
  "batchSize": 4,
  "saveEvery": 10,
  "processes": 2,
  "minSlices": 1
}
```

这是单个路线的配置片段，放入完整 JSON 的 `pipelines.rvc`，再让 `DSH_ADVANCED_CONFIG` 指向该文件。路径只是示例。也兼容顶层 `rvcRoot/rvcRepo/rvcEnv` 与环境变量 `DSH_RVC_ROOT/RVCX_REPO/RVCX_PY`。

RVC 引擎需包含：

- `train/preprocess.py`、`train/dataset/extract_f0.py`、`train/dataset/extract_hubert_feature.py`、`train/train.py`、`train/train_index.py`。
- 对应采样率的 `assets/pretrained_v2/f0G48k.pth` / `f0D48k.pth`（40k 则换成对应 40k 底模）。48k 配置 `configs/v2/48k.json`；40k 沿用该引擎的 `configs/v1/40k.json`。
- 工程匹配的 HuBERT / ContentVec。检查兼容 `assets/hubert_base/` 的 Transformers 配置与权重布局，或 `assets/hubert/hubert_base.pt` 的旧布局；实际加载方式由该引擎实现决定。
- `logs/mute` 中对应采样率音频、v2 的 768 维静音特征和两种 F0 静音文件；rmvpe 还需 `assets/rmvpe/rmvpe.pt`。
- 引擎自身的训练依赖、可用 CUDA；检查器另核对 torch、numpy、soundfile、faiss、librosa、parselmouth 和子进程 PATH 中的 FFmpeg。

这些环境与权重不随插件分发，也不会在加载插件时安装。`minSlices` 是防空产物的技术门槛，默认 1 不表示一条语音足够训练。正式语料数量、时长、覆盖与训练参数需按任务检查。

## 在统一入口执行

先查询 `adv_train_catalog route=rvc`。操作参数以返回 schema 为准。以下示例中的工具均为 `adv_train`：

```json
{"route":"rvc","action":"doctor","params":{"deep":true}}
```

```json
{"route":"rvc","action":"scan","params":{"input":"D:/AudioWork/角色A"}}
```

确认角色、清晰度与句尾后，可用 `adv_tail_audit` 检查原始目录。随后：

```json
{"route":"rvc","action":"build","params":{"input":"D:/AudioWork/角色A","expName":"role-a"}}
```

```json
{"route":"rvc","action":"status","params":{"expName":"role-a","jobId":"<build 返回的 jobId>"}}
```

等 `state=succeeded` 后，按顺序执行：

| action | 工作 | 成功核查 |
| --- | --- | --- |
| `preprocess` | F0、v2 HuBERT 特征、训练清单 | 每条音频的特征配对、非空/有限数值、768 维、单角色清单与静音项 |
| `run` | GPU 训练；可传 epochs、batchSize、saveEvery | 本实验导出的小模型达到目标 epoch，模型格式/采样率/权重有效，存在 G/D 检查点 |
| `index` | 构建检索索引 | 读取 `added_*.index`，768 维且向量数大于 0；`trained_*.index` 只是中间产物 |
| `register` | 归档音色包 | 模型、索引、对应配置、训练清单与训练日志及 SHA-256 |

`preprocess/run/index` 也都需要 `expName=role-a`。前三个为后台操作，每次都使用该次返回的 `jobId` 查询 `status`，等待成功再推进。省略 expName 使用配置值；省略 jobId 查询该实验最新任务。只得到启动回执不能当作完成。

```json
{"route":"rvc","action":"run","params":{"expName":"role-a","epochs":150,"batchSize":4,"saveEvery":10}}
```

```json
{"route":"rvc","action":"register","params":{"expName":"role-a","role":"角色A","outDir":"D:/AudioWork/voice-models/role-a-v1"}}
```

默认要求成品索引。明确不使用检索索引时，可跳过 index 并为 register 设置 `includeIndex=false`；清单会记录索引为空。音色包固定标记 `qualityAccepted=false`，没有执行推理或试听验收。

## 后台任务、续训与保护

工作记录位于 `<workRoot>/rvc/<expName>/`，每次任务保存独立 jobId、请求、状态和停止标记；阶段日志保存在该实验目录。RVC 原生产物仍位于引擎 `logs/<expName>`，小模型位于 `assets/weights`。统一调用回执另由主入口保存。

同一引擎实验由锁保护，避免本入口同时运行多个阶段；它不会接管已有外部训练进程。实验名只允许英文、数字、下划线与短横线，最多 64 字符。请为每个角色与数据版本使用新实验名，不与 WebUI 或其他训练任务共用实验。

已有出料仅当 `build resume=true` 且原始语料 SHA-256 指纹一致时复用。已有模型或 G/D 检查点会拒绝普通 run；续训需显式 `resume=true`，且必须有编号和内部轮次对应的最新成对 G/D、优化器状态、配置与清单。只有导出的小模型不等于具备续训材料。

0.3.1 会先复制到 `<workRoot>/rvc/<expName>/checkpoint-backups/<UTC时间>-<jobId>/`，要求该位置位于引擎 `logs` 之外，保存 G/D、config.json、filelist.txt 和 backup.json。备份重新核验大小、SHA-256 和源文件稳定性，确认后才启动训练；空间不足、文件不配对或无有效轮次时直接失败。每次备份独立保留，不覆盖、不自动清理旧备份；大型 G/D 需要预留相应空间。

续训从已核验的快照读取，严格加载 G、D 及两份优化器状态。不会调用上游的“恢复失败后加载预训练底模”兜底；任一加载失败就结束该任务。120 秒内没有收到完整恢复证明也会停止。status 显示备份目录、检查点轮次、目标总轮次和“恢复已确认/待确认”；该次 jobs 目录保留 resume-proof.json。epochs 必须大于已有轮次。原生引擎可能重跑检查点所对应的一轮，恢复证据中的 fromEpoch 是已加载状态的轮次。

为避免修改已安装引擎，本入口冻结该次 train/train.py 入口到任务目录并装载严格恢复钩子；只接受已检查的 utils.load_checkpoint / latest_checkpoint_path 调用布局。其他分支会拒绝无保护续训，不能据此宣称兼容所有 Applio/RVC 版本。已有音色包拒绝覆盖，更新模型应指定新的 outDir。

```json
{"route":"rvc","action":"run","params":{"expName":"role-a","epochs":200,"resume":true}}
```

### Applio 界面提醒

收到过“点开始训练后旧 G/D 被清理，实际从第 1 轮重跑”的反馈，未取得该次安装的版本与选项记录。核对 Applio 提交 [fb427aa](https://github.com/IAHispano/Applio/tree/fb427aa388a954a387da7034fbbc08313f7ffe8c)：[界面的 Fresh Training 选项](https://github.com/IAHispano/Applio/blob/fb427aa388a954a387da7034fbbc08313f7ffe8c/tabs/train/train.py#L713) 默认关闭；打开后，[训练入口会删除旧 G/D 等产物](https://github.com/IAHispano/Applio/blob/fb427aa388a954a387da7034fbbc08313f7ffe8c/rvc/train/train.py#L278)。不能将其概括为所有版本每次点击开始都会删除。

在 Applio 界面继续训练时，先把成对 G/D、配置与清单另存到 logs 外，核对 Fresh Training / cleanup 关闭，检查加载记录与起始轮次后再长时间运行。本插件的自动备份只在本插件 `adv_train rvc/run resume=true` 入口执行；它不会拦截外部 WebUI 按钮，也未新增完整 Applio 训练后端。

停止操作只给本入口任务发送停止请求，由该任务结束自己的子进程树；之后仍需查询同一 jobId 确认 `stopped`。未知任务不会按进程名搜索并结束。中断后用新的阶段任务继续，损坏或不完整产物会被检查器拒绝。

Windows 后台阶段使用独立 Job Object，协调进程异常退出时由系统结束它拥有的子进程树，避免遗留训练进程。状态读取结合心跳与进程探测：短暂探测异常有 10 秒缓冲，失联或过期心跳返回 lost / unknown，不当作成功；这些状态需先核查日志再重试。

```json
{"route":"rvc","action":"stop","params":{"expName":"role-a","jobId":"<需要停止的本入口 jobId>"}}
```

本路线止于训练与模型交接，不暴露上游脚本的 all、infer、convert 或翻唱操作。验证证据与限制见 [VALIDATION.md](VALIDATION.md)。
