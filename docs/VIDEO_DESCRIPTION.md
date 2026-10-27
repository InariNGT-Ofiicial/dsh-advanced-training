# 视频简介

想把游戏里的角色语音、番剧中的台词，或自己录制的声音，训练成能继续使用的音色模型？dsh-advanced-training 是 DeepseekHarness 中面向音色模型训练的统一入口，把素材接入、数据整理、质量检查、模型训练和产物管理接到同一条流程中。它整合游戏解包、番剧摘句，以及 DDSP、GPT-SoVITS、RVC 三条训练路线，也支持从已有的单人语音集开始。

主插件负责组织步骤、检查前置条件、调用对应工具，并管理训练状态、模型配置和日志；具体训练算法仍由各自的上游工程执行，不同路线的模型分别保存。本期功能范围止于音色模型训练与产物交接，配音、翻唱和成品发布另行展开。

主仓库：[dsh-advanced-training](https://github.com/InariNGT-Ofiicial/dsh-advanced-training)

接入来源：[dsh-train-pipeline](https://github.com/InariNGT-Ofiicial/dsh-train-pipeline)、[dsh-anime-voice](https://github.com/InariNGT-Ofiicial/dsh-anime-voice)、[rvc-voice-kit](https://github.com/InariNGT-Ofiicial/rvc-voice-kit)

适用版本：0.4.0。RVC 与 Applio 已接入单角色训练、显式续训、索引生成与音色包交接；Applio 可在插件内接管既有实验，自动备份 G/D 并核验模型、优化器和 AMP 恢复，无需外部界面点击开始。合成语料实际验证不代表真实角色音质验收，具体范围见 VALIDATION.md。
