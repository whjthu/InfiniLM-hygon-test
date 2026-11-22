# Z-Image 目标 tokenizer 检查

从目标端已封存的四个 tokenizer 文件读取供应方定义，再按原样编码 12 条已绑定的训练提示。校验直接批量调用与逐条调用的 token ID 和 attention mask 一致。输出仅保留长度及固定字节序的 ID 摘要。

原单步使用父作业 `1087656`、节点 `f11r4n14`、4 核、8 GiB、零张卡及现有 DTK 容器。容器把精确 tokenizer 目录、3601 字节训练 CSV 和官方资产清单分别只读绑定到参数指定的位置。设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，关闭 GPU 可见性。

```sh
python3 -B -u probe.py --tokenizer /tokenizer --metadata /training.csv --manifest /manifest.json
```

检查范围是 Hugging Face tokenizer 的离线输入路径。DiffSynth 模板及 A100 token 对照由原配方分别验收。读取原 stdout 时，先按保存的 SHA 校验整份日志，再提取唯一 JSON 行。
