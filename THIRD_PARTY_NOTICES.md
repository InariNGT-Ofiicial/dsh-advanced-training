# 内置来源与分工

整合复用现有项目的运行源码，不复制虚拟环境、模型权重、游戏/番剧素材或本机配置。

- `vendor/ddsp-pipeline` 来自 [dsh-train-pipeline](https://github.com/InariNGT-Ofiicial/dsh-train-pipeline)，为 MIT；完整 [LICENSE](vendor/ddsp-pipeline/LICENSE) 与原声明保留。
- `vendor/anime-voice` 来自 [dsh-anime-voice](https://github.com/InariNGT-Ofiicial/dsh-anime-voice)，包含分析脚本和 tts-train-pipeline。原项目未授予开放许可、标记 UNLICENSED，本次不改写该声明。
- DDSP-SVC、GPT-SoVITS、MSST、ASR、声纹模型和解包工具分别遵循各自许可，由使用者独立准备。

来源提交及每个冻结文件的 SHA-256 见 [manifest](vendor/manifest.json)。上游原文档反映独立项目能力，主入口操作以 adv_train_catalog 为准；推理/翻唱/发布不暴露。

维护者可用 scripts/snapshot_backends.py --ddsp <本地仓库> --anime <本地仓库> 从明确的本地 Git HEAD 更新快照。更新后复核 diff、哈希和测试，不复制未跟踪的私人配置或素材。
