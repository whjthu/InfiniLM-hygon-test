# Gemma 的真实本地输入前提

复用已有且经审计的 USGS v2 数据，不GET、不重绘、不改现有冻结包。调用已冻结统一五VLM数据适配器，实际生成Gemma的72训练记录（user+assistant FINAL）及36开发记录（仅user）；原数据和媒体不进Git。独立审核逐行join、完整题面、源组/事件跨split隔离元数据、18张train/dev PNG的全bytesSHA/尺寸/各chunkCRC；不打开开发答案或任何盲测正文。

运行入口为固定Python执行 once.py --prepare-local。一次独占本地namespace、60秒原epoch、实际WindowsJob2CPU/2GiB。重复自动补位调用只消费原终态和新输入hash，不再次生成或接远端。refill-candidate-private.json仅作为root的下一必要nativeCPU候选输入，不能作为训练READY。

Gemma原512/SDPA/GC OFF约束保留；原生全长度、EOS、mask、有效监督token、目标初值和集群认证均未在本机工具中解决。没有模型/原生encoder/optimizer、Slurm或SSH调用。当前认证由root报告失败，目标目录只是声明路径，并未部署。只有真实目标前提、发布claim和fresh准入通过后才能训练。

图源是USGS政府公开数值的原已冻结派生图，并非USGS官方图。盲组这里只读manifest中的分组/事件身份元数据，不重认证其跨其他模型的暴露状态，不宣称新鲜盲测或领域能力。
