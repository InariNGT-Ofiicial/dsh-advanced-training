# Changelog — dsh-advanced-training

## 0.2.0 (2026-10-26)

- 新增统一入口 adv_train_catalog / adv_train_plan / adv_train。
- 固定版本内置 DDSP、GPT-SoVITS 管线和番剧素材脚本，记录来源提交、哈希和许可。
- 接入游戏、自备语音、番剧流程；两类模型调用原训练后端，格式不合并。
- 补齐番剧 file / train.list 到训练器 voice_path 的台词桥接。
- 增加统一配置、参数校验、执行回执和错误路径保护，不暴露推理/翻唱/发布。
- 保留四个原有质量工具；RVC 本版未接入训练编排。
- 新增契约/集成与合成语料出料验证，不把这些检查等同于完整 GPU 训练验收。

## 0.1.0 (2026-10-26)

初版。把三条实测有效的「先进训练方法」收拢成独立 DSH 插件（独立复刻，不动上游插件）：

- `adv_doctor`：三线工程体检（DDSP / GPT-SoVITS / rvc-voice-kit）
- `adv_tail_audit`：三线通用句尾体检，口径与 dsh-train-pipeline 0.5.0 的 `end_metrics` 完全一致
- `adv_unk_patch`：GSV 消 UNK 文本补丁，复刻自 dsh-tts-train-pipeline 0.2.x
  （幂等 / `.bak_dsh_unk` 备份 / revert / `ast.parse` 语法门 / `already-external` 识别 / 符号表核对）
- `adv_ab_eval`：无参考验收 A/B（新开发）——ASR 回转写 + CER + 末尾字覆盖 + 尾音统计；
  默认离线用本地 whisper 缓存；自动补 ctranslate2 的 CUDA DLL 搜索路径

验证：契约自检 4/4；芳乃留尾数据集体检（−23.4dB / cut 0.017）；GSV 旧切尾数据集
体检（cut 0.642，抓到待重出料）；补丁四态（apply/幂等/revert/diff）；ASR+CER 8 条冒烟。
