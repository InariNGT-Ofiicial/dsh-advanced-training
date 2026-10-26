# rvc-voice-kit

用 RVC 做**音色转换（SVC）**的一套命令行工具链：训练编排 → 批量换声 → 客观评测。

不是 WebUI 的封装，也不重造模型。它解决的是 WebUI 之外的三件事：

- **训练能脚本化、能续跑**：六段流程每段校验产物，失败自动停在原地，不长跑一晚上发现第一步就错了。
- **换声能批量、不踩坑**：`rvcx.voice` 一条命令处理整个目录，参数从模型清单自动读取。
- **结果能判断**：音色位移、可懂度、声调保真都有量化口径，并且写清了哪些指标会说谎。

> 已验证版本：RVC 2.3.260718（上游 commit `81eed5e`）、Python 3.12、torch 2.7.1+cu128、
> RTX 5060 Laptop 8 GB（sm_120）。48k / v2 / batch 4 在 8 GB 显存内可跑。

---

## 30 秒上手

```bash
# 1. 拿上游引擎 + venv + 依赖（约 12 GB，只做一次）
powershell -ExecutionPolicy Bypass -File setup.ps1 -Root E:\RVC

# 2. 告诉它你的路径（复制模板，改 3 行必需项）
cp rvcx.local.example.toml rvcx.local.toml

# 3. 体检
E:\RVC\venv\Scripts\python.exe -m rvcx.voice doctor

# 4. 换声
E:\RVC\venv\Scripts\python.exe -m rvcx.voice convert ^
    --voice 我的角色 --input dry/ --output wet/
```

`doctor` 会逐项告诉你缺什么、怎么补。它检查的正是最容易卡住的四处：
上游代码、venv、rmvpe 权重、**ffmpeg 是否真的在子进程 PATH 里**（RVC 代码写死
`cmd=["ffmpeg"]`，不在 PATH 时的报错看起来很像「输入文件损坏」）。

---

## 目录结构

```
rvcx/
  paths.py          所有外部路径的唯一出口（环境变量 > rvcx.local.toml > 默认）
  voice.py          换声入口：convert / list / doctor
scripts/
  dataset/          语料生成（声韵母覆盖均衡）· 批量合成（edge / CosyVoice，含写盘前 dtype/峰值闸）
  train/            p1_run.py 六段编排 · launch_train.py 脱离进程 · p2_launch.py · status.py 进度 · extract_small.py 导出
  eval/             sweep.py 参数网格 · compare_ddsp.py 对拍 · ltas_check.py 音色位移 · ab_make.py 试听
  dialogue/         台词实验：TTS 源 → 换声 → 指标 → ASR/CER → 对照页
  service/          常驻服务：svc_tts（TTS）· svc_rvc（换声）· svc_openai_shim（OpenAI 兼容门面）
  bench/            延迟基准：TTS / RVC / SILK / 端到端
  diag/             疑难排查（f0 估计器对比等）
models/
  INDEX.md          模型索引（人读）
  index.json        模型索引（机读，含 sha256）
  measurements/     原始测量数据（检查点横评、DDSP 对拍）
packages/
  dsh-plugin-voice-say/     DSH 插件：voice_say 工具（文本 → 目标音色 wav）
  astrbot_plugin_voice_patch/  AstrBot 插件：语音回复
docs/
  RVC换声插件方案.md  完整技术报告：源码分析、架构建议、全部实测
  判据手册.md        每个指标能证明什么、不能证明什么
  踩坑清单.md        RVC 2.3 CLI 化的 23 个坑
  模型清单格式.md    .rvc-manifest.json 的三种 schema 与归一化
tools/
  build_model_index.py   扫音色库重新生成索引
  dl_multi.py            多线程分段下载（单连接被限速时用）
requirements/
  eval.txt          评测侧依赖（单独 env）
setup.ps1           一键拉取上游 + 建 venv + 装依赖
```

---

## 常驻服务：把「文本」变成「目标音色语音」

接入聊天端（dsh-qq / AstrBot）时，**别让每条回复都起一次进程**——模型加载占了大头。
拆成两个常驻 HTTP 服务，模型只加载一次：

```
:8788  svc_tts.py     TTS      （跑在 TTS 那个 env）
:8789  svc_rvc.py     换声     （跑在 RVC 那个 env）
```

两者的 env **不能合并**（TTS env 没有 faiss/rmvpe，RVC env 没有 cosyvoice）。

```bash
# TTS 服务
<TTS-env>/python scripts/service/svc_tts.py --port 8788 \
    --src-root <CosyVoice 源码目录> --prompt <参考音色 wav>

# 换声服务（RVC env）
<RVC-env>/python scripts/service/svc_rvc.py --port 8789 \
    --model <model.pth> --index <added.index>
```

调用：

```bash
curl -X POST 127.0.0.1:8788/say -H 'Content-Type: application/json' \
     -d '{"text":"你好"}' -o tts.wav
curl -X POST 127.0.0.1:8789/convert -H 'Content-Type: application/json' \
     -d '{"in":"tts.wav","out":"rvc.wav","index_rate":0.75,"pitch":0}'
```

`svc_openai_shim.py` 把同一条链包成 OpenAI 兼容的 `POST /v1/audio/speech`，
AstrBot 的内置 `openai_tts_api` provider 可以直接指过去，**不需要写 AstrBot 插件**。

两个必须知道的坑：

1. **faiss 打不开含中文的 index 路径**——只打 traceback 然后**静默按 index_rate=0 跑**，
   表现为「换声质量莫名下降」。`svc_rvc.py` 启动时会把模型/索引复制到 ASCII 目录来绕过。
2. **产物必须落在 agent 的 cwd 里**——dsh-qq 的 `sendFile.restrictPaths` 只放行
   `media` 目录 + `config.cwd` + `extraRoots`，生成到别处会被工具直接拒绝。

---

## 两套环境

换声主链和评测链要分开装，合在一起会互相顶版本（尤其 numpy 和 ctranslate2）：

| | 用途 | 关键依赖 |
|---|---|---|
| **换声 env** | `rvcx.voice`、`scripts/train`、`scripts/eval`、`dialogue_metrics*` | torch 2.7.1+cu128、faiss-cpu、librosa |
| **评测 env** | `dialogue_cer*`（ASR 算 CER） | faster-whisper 1.2.1、pypinyin 0.55 |

换声 env 由 `setup.ps1` 建好；评测 env 见 `requirements/eval.txt`。

---

## 实测结论

以下数字来自本仓库 `models/measurements/` 里的原始文件，可用 `tools/build_model_index.py` 复算。

### 轮次不是瓶颈

同一角色、同一参考集、同一段干声，只改检查点（单片段，index_rate 0.75）：

| 检查点 | e20 | e40 | e60 | e80 | e100 | e120 | e140 | e160 | e180 | e200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Δr | .0440 | .0438 | .0439 | .0445 | .0430 | **.0449** | .0433 | .0424 | .0440 | .0426 |

**e20 就进了平台区**，之后全在噪声里抖动。200 轮花了 4h21m，相对 120 轮拿不到可判差异（听感略好，客观不可分）。

⇒ **新角色 100–150 轮封顶**，留两个检查点各一份。要质感取 200，要性价比取 120。

### RVC vs DDSP

同一段干声、同一参考集：

| 系统 | Δr（朝目标移动） | 精确归零比例 |
|---|---|---|
| DDSP-SVC 20000 步 | **−0.0189**（三句一致为负，反而远离目标） | 1.4 % |
| RVC e120 | **+0.0409** | 0.0 % |

RVC 的 `exact_zero_ratio = 0.0` 意味着**不需要补气口工序**，这是实用上的明显差别。

### index_rate 在台词上没有收益

| index_rate | 0.0 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| Δr | .0398 | .0404 | .0409 | .0411 |
| 中文台词字数差异 | — | ≤2 字 | ≤2 字 | — |

索引库是用**日语**数据建的，理论上有「把中文音素拉向日语分布」的风险；实测没有。
台词场景可以直接 `--index-rate 0`，省掉索引依赖。

### 跨语言（日语训练的模型换中文）

| 系统 | Δr | 拼音 CER | 声调 CER | 声调轮廓 r |
|---|---|---|---|---|
| 源（CosyVoice2 中文） | — | .010 | **.000** | 1.000 |
| 芳乃 RVC e120 | +.2141 | .017 | **.000** | .985 |

声调一个字都没错。但——

> ★★ **声调保住靠的是 `pitch 0` 这个工程约定**（f0 直接由源音频驱动），
> 不是模型懂汉语。启用 pitch shift 变调时，只能保住轮廓、丢掉音区；
> **中文唱歌必须单独验证。**

---

## 用 TTS 音色造一个 RVC 模型

手上有的是一个 **TTS 音色**（例如某个克隆音色），只有十几秒参考音，
但想要一个能换声的 RVC 模型——这条路走得通，关键是**先用 TTS 批量造语料**。

```bash
# ① 造语料：按声韵母覆盖 + 句长分布生成句子表（避免全是枚举句）
python scripts/dataset/gen_corpus.py --out E:/AI-RVC/corpus/myvoice --sentences 300

# ② 批量合成：edge / CosyVoice 双后端，写盘前有 dtype+峰值+静音三道闸
python scripts/dataset/synth_corpus.py --backend edge \
    --texts E:/AI-RVC/corpus/myvoice/text.txt --out E:/AI-RVC/dataset/myvoice

# ③ 训练（先跑数据链，再训）
python scripts/train/p1_run.py --exp myvoice --data E:/AI-RVC/dataset/myvoice \
    --steps prep,f0,feat,list --f0method rmvpe
python scripts/train/launch_train.py --exp myvoice --data E:/AI-RVC/dataset/myvoice \
    --epochs 150 --steps train,index,infer
```

三条必须记住的：

1. **参考音 14 秒不够训模型，但够做 TTS 的参考音。**
   所以第一步是用 TTS 把语料「扩」到 20 分钟以上，模型训的是这个音色。
2. **写盘前必须查 dtype 和峰值。** TTS 返回的常常是 int16 量级的数组，
   按 float 写 PCM_16 会硬削顶，谐波污染 HuBERT ⇒ 换声后 CER 直接 1.0。
   `synth_corpus.py` 内置了这道闸。
3. ★ **同源自测会骗人。** 训练集和测试集都来自同一个 TTS 时，
   它们共享同一个带宽指纹（24 kHz 源在 15 kHz 以上是数字静音），
   Δr 会虚高十几 dB 的量级。**交叉验证集必须用另一个引擎合成**，
   并且必须做一次真人干声通道的测试。

完整方案与实测见 [`docs/DS娘RVC模型制作方案.md`](docs/DS娘RVC模型制作方案.md)。

---

## 判据：三个会说谎的指标

完整版见 [`docs/判据手册.md`](docs/判据手册.md)，这是最短版本：

1. **Δr 只表示「移动了多远」，不表示「像不像」。**
   它的上限被录音信道锁死：同一录音链的源本身 r 就到 .96，Δr 只能在 0.04 里动。
   跨信道（TTS vs 真声）比较 Δr **没有意义**。每次对拍都要写参考集自测上限。

2. **LTAS 相关判音色方向可靠，判自然度不可靠。**
   MFCC 类指标已从判据里剔除——它分不清「像源」还是「像目标」。

3. **声调必须换测量工具复验。**
   `pyin` 在换声输出上会八度跳变（半音 std 2.94 → 6.92），算出 r=.44 的假警报，
   而同一条音频的声调 CER = 0。两个判据打架时，**先怀疑测量工具**。
   现用 `rmvpe` + 线性对齐（时长比差 ≤1.2% 时根本不需要 DTW 弯曲）。

---

## 它不做什么

- **不带权重**。`models/` 里只有索引与 sha256 校验和。单个 `added_*.index` 约 170 MB，
  超 GitHub 单文件限制；且音色模型是衍生作品，公开分发需自行解决授权。
- **不带音频**。实验用的台词与歌声属各自权利人，仅作研究性评测。
- **不做实时**。这是离线批处理工具链，延迟与流式不在范围内。
- **不替你判断听感**。客观指标只能排除明显坏掉的输出，最终仲裁永远是耳朵。

---

## 许可

代码 MIT。上游 RVC 亦为 MIT。权重、音频、角色不在此授权范围内，详见 [LICENSE](LICENSE)。
