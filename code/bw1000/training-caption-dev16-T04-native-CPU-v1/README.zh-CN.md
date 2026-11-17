# 固定dev16的T04原生CPU预处理：仅准备

本scope只新16图原生推理encode/collator，未训练/生成/评分。原350框架b641edb0/原A100 env-vlm（Torch2.8.0a0+5228986c39.nv25.6、Transformers5.12.1、Swift4.4.2）的原T04 load_model=False和native collator API已有原64 train-mode CPU资格。新增set_mode(transformers)、labels None、无答案dev16不得继承原train64；只能新实际CPU验证。共享base inference API来自已存在Qwen holdout源码，具体T04模式仍待实证，不改模型数学或原17项recipe。

get_model_processor只指newnamespace model-metadata-only视图；此视图精确14metadata，总来源原A100模型根，逐SHA、exact inventory、offline和own HF cache，无4shard/initial.pt。原物理source路径保provenance，新view路径纯CPU输入资格。Python audit真实拦截4shard+project.pt open探针，Torch/safetensors load API封锁；这不是任意C++原路径访问的内核级证明，主要无weights边界为only-metadata view。unexpected lookup/网络/device触碰即fail closed。没有原Model load/constructor/forward/LoRA/optimizer。

原fixed16 JPEG2,492,125字节的原SHA/尺寸取自root真实独审9bec，不重下载或重读原图；私有DATA_MANIFEST SHA5b758107194df4f2599cad3d2598fb4f012eb220c3a75b1052db4d64eac3fcd1，inputs.jsonl SHA8628ea546b2ef009e1ec6b7bd83a58e2a57617c10970ba27303852a363642977。私有准备目录local-only/training-caption-dev16-T04-native-CPU-v1-input-preparation；仅由stdlib读取三份小冻结metadata生成，无nativeCPU/图片/reference读取。实际未来staging仅16JPEG+inputs.jsonl+DATA_MANIFEST共18datafiles，不含80条reference；媒体身份/路径清单不入Git。

完整prompt固定<image>+Describe this image in one short sentence.，模板implicit llava_onevision1_5、max_length4096、max_pixels262144、右padding、非paddingfree、truncation raise。每行真实mode/labelsNone/imageTokens/完整长度/全字段finite/hash/CPU/provider逐槽记录，16全分母失败不丢；仅private hash/长度/计数，不保存raw处理张量。此CPU资格只T04，不等于Qwen3arm预处理、CIDEr或通用效果通过。

receive_once.py复用qualified owned-process supervisor：parent单epoch300、inside260、supervisor270、TERM5/KILL5及terminal child.wait/PGID/startticks；actual2CPU排除0/1、Torch2、groupRSS8GiB采样停止阈值（不是commit/RSS硬峰保证）。不新Slurm/GPU。旧qualified350为只读provider，运行时逐350清单校验，不复制/新造框架。新namespace /home/whj/nvidia-baseline/alignment-assets/training-caption-dev16-T04-native-CPU-v1，existing拒绝。

根发布前0remote/CPU/HTTP/Git。未来root仅通过现有授权transport一次注入CLAIM(本次pub)、REQUEST(唯一ID)、DATA_MANIFEST_SHA256和FILES(精确bytes/sha/b64)执行receive_once；已接入本次发布的 launch_once.py；准备阶段未调用。runtime src exact5files：check_native.py/MODEL_METADATA.json/ORIGINAL_SOURCES.json/FRAMEWORK_FILES.json/SOURCE_PINS.json，data exact18files。所有body都由pub/私有manifest固定，root应先核新namespace及资源状态。unknown不重播。源码准备只AST，无新旧测试重跑。

原oEmbed16失败继续保留，license/署名仍pending；内部CPU工程可独立推进，不声称redistribution-ready。Dataset_GO/training_GO/GPU_GO与正式17验收均false。

根唯一CLI（先发布后调用，原client epoch含Git准入/固定16图上传）：

`python -B -X utf8 launch_once.py --publication <40hex> --source-mf-sha256 <MF> --claim-sha256 <CLAIM>`

实际入口核published MF/CLAIM/own clean/每sourceSHA、私有18data成员完整SHA；原DATAmanifest和16JPEG不入Git。固定local-only/training-caption-dev16-T04-native-CPU-v1-original-dispatch独占，exec-intent保bodySHA与实际attempt字段；UNKNOWN不重播。来源8GiB仅RSS采样停止阈值，CPU-only新mode资格待真实终态。
