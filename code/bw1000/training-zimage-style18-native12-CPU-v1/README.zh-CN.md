# Z-Image Style18 新目标原生12图CPU输入资格

固定新request `4ce20eb249de478a9f2c849b28f0db93`，新RUN `R/results/training-zimage-style18-native12-CPU-v1`。所有GO=false，根审/发布并采纳后唯一执行。候选parent1087656/node f11r4n13、4CPU/8GiB/0GRES，原epoch300s含准备和传输，native work到原epoch+220、Slurm最多240，剩余不足必须缩水且保清理预算，不重置epoch。根使用现有held/终态accounting/owned cleanup包装；本包没有SSH/Slurm launcher，不创建新通道。

## 已有目标凭证与去重

旧183已通过CPU依赖检查；184因imageio失败；185补依赖成功；186真实native-meta/PEFT成功。它们不是Style18的native12像素/提示资格，不重跑任何一个。新backlog `zimage-usgs-style18-exposure25-resume20.target_image_preflight.target_native_result` 尚pending；原18图CSV、两constructor和官方PIL参考仅本机证据，不能冒目标12PASS。

目标原40个DiffSynth源在 `R/results/training-zimage-target-native-meta-CPU-v1/source/upstream`；7964B provider manifest cc0aa379…证明确实包含data三文件：__init44B/9118710b…、operators7735B/c02a73eb…、unified_dataset4972B/b6e0729f…。与已发布 `tools/preflight_zimage_style18_train_input.py` 的provider pin完全相同。不是缺DiffSynth安装，也不需要重新下载。

原186复用依赖在 `R/results/training-zimage-target-native-meta-CPU-v2/dependencies/site`（2852文件445245B manifest50d3934b…，pandas2.2.3/imageio2.37.0等）；公共运行库在 `R/results/training-t06-sdxl-runtime-v1/site`。容器是既有 `R/containers/dtk26-sandbox-r1`。这是旧资格身份，本次仍核原40源与实际data provider，记录实际Torch/TorchVision/Pillow及原生输出；不据旧metadata称新目标完成。

## 现有12图和精确CSV

`prepare_inputs.py`已真实本机运行：用已发布Gemma INPUTS与T07 binding逐ordinal核12幅图片的bytes/SHA，源为 `R/results/training-gemma-usgs-render72-native-CPU-v1/data/train/media/000..011.png`。这与本轮根实际核过的Gemma源图相接，不复制/重采样图片。本机新验证只读公开metadata，没有打开目标图片正文。

它从原已发布prepare.py的CAPTION字面量、同csv.DictWriter和12固定路径严格重现旧CSV；3601B/SHA `f71b51feb39af32e1d79b7750c9c4c437d930a35f69b60e52f80389ba6084350`，不存在答案选样或新数据。私有CSV为 `local-only/training-zimage-style18-native12-CPU-v1-input-preparation-original/metadata.csv`；根可读共享交付副本 `local-only/training80-user-all-nodes-live-first/ZImage-style18-train-f71b51.csv`，同pin。提示原始正文不入Git，不读取旧私有CSV/gold。

根先独占建新RUN，把本包exact源放RUN/source、CSV放RUN/data/train/metadata.csv，创建空RUN/data/train/images和RUN/evidence；`evidence/native`须不存在。图片目录只读嵌套bind到 `/zimage-input/train/images`，父RUN/data只读bind `/zimage-input`。原始12PNG与3601CSV必须在原epoch内fresh核hash；目标原生入口还会在实际读样本前后核一遍，不把此处metadata映射称目标bytes复核。

## 实际调用

PLAN.json包含固定bind、环境和完整Apptainer argv模板。使用已成功的绝对Apptainer，无`--rocm`；`/usr/local/hyhal→/opt/hyhal:ro`只有在根核过允许的原系统alias关系后使用，不创建别名。三GPU可见性均-1、0GRES、离线模式。PYTHONPATH严格为 `/native-source:/zimage-runtime:/training-runtime/site`，避免import落入另一份DiffSynth。

实际容器Python调用为：

```
python3 -B -u /input-stage/run_native.py --epoch ORIGINAL_EPOCH --deadline ORIGINAL_EPOCH_PLUS_300 --output /evidence/native
```

新worker从原epoch+220设alarm，固定request/node/step/4CPU affinity，fresh逐hash核原40源，调用原未改8645B入口（本包source/tools exactcopy）。该原入口使用 `UnifiedDataset.default_image_operator(height=256,width=256,height_division_factor=16,width_division_factor=16)`，逐12原生BILINEAR/center-crop/PIL RGB256和提示SHA比较冻结参考。输入是12图CSV，不是tokenizer编码；VAE/Turbo融合/权重/optimizer不进入。审计hook拒绝tensor文件，is_initialized只被动检查；库加载的隐式GPU副作用不冒SDK完整0证明。任何失败保12分母和私有why，原caller进度/owner供根独审与清理。成功只为输入资格，不是训练READY或正式17。

数据沿原USGS公开政府数值、本项目衍生图和固定train/dev/盲测closure；只train12，dev6及盲测均不读。来源：[USGS copyrights and credits](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)。原T07 COCO20、Flash2设备能力、融合结果、初值/数值与25+5恢复全部独立保留，不能由本CPU包晋级。
