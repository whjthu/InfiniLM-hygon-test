# T03 缺失的 HIP train72 输入资产

复用已下载且冻结的 USGS render144-v2，仅训练72问/12图：原始 train inputs 与 labels 通过共同 `training-domain-data-registry-v1/prepare_vlm.py:join` 同序合并为完整 user/image + assistant FINAL，不增加字面EOS、不缩短问题、不降图像尺寸。图像保持原PNG512×512字节并逐chunk核CRC。新的绝对图像路径指向 Hygon 专属 namespace，原文件/其他模型资产不改。

这不是72个新独立地震样本：它是原冻结训练划分向 T03/HIP 的必要平台绑定。原24图/144QA与train/dev/blind closure隔离只按manifest元数据核，dev/blind题面、标签、图片均未打开。没有读取模型/初始参数、调用Torch/编码器或使用网络/SSH/Slurm/GPU。

固定本机请求 `c3dfa76b1764440e8cc8b9dd806cb46a`，`prepare_once.py` 只运行一次，独占 ignored `local-only/training-internvl-USGS72-HIP-input-asset-v1-original`。输出14个待部署DATA文件：train-swift.jsonl、data-binding-private.json、12PNG；只有文件bytes/SHA进入公开小清单，问题/答案/事件清单/图片均留private。

## 必须分开的资格

- A100 的 T03 render144原native144已通过，但不是Hygon新72输入资格。
- Hygon原CPU360/v5实际通过original64/meta504；原worker硬编码64和旧reference，不能把本包改个dataset路径直接重调它。本包未调用它。
- `NATIVE72_CONTRACT.json` 只是新CPU最短必要接口，尚非完整CPU运行包。必须以新的claim/epoch/source注册、派生72数据相关检查和独立reader，保留模型provider/13metadata/原HF basename/SDPA/语言GC+和完整4096，不继承旧64值指纹。新CPU需实际full-vs4096 encode144次及collator72次、原生监督EOS/mask、图像字段与selected-logits，缺项计72完整分母。
- namespace路径声明不证明目标文件存在。目标模型权重、原rank initial/RNG、设备/optimizer真更新和新进程恢复仍是后续独立条件；Model/训练/GPU/CPU GO及READY都false。不得重传未知旧模型请求或重播任何旧CPU/GPU句柄。

本包只产生新平台输入资产，不能称已训练/新领域能力/正式17项通过。图由本项目据USGS公开数值渲染，不能冒官方原图；原署名和来源为 [USGS Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)，许可边界沿[USGS版权与署名政策](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)及冻结源资格，不重新获取数据，也不包含第三方照片。
