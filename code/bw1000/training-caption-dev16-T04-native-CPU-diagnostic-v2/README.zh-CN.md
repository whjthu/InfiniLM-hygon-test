# T04 dev16 原生 CPU v2：仅诊断候选

原 v1 请求 b1f1e5b3dd884a3093273b1d99c54091 已真实 FAILED，0/16，不能改写。只有总 phase 和 ValueError，无法在 get_model_processor/get_template/set_mode/provider/首槽 encode/fingerprint/collator 间确定原因。Qwen 已有同 mode 源不证明此次 T04 具体错误。v2 不猜修、不放宽任何原数学或门禁。

check_native.py 增加38个固定 need-code、细分 phase、provider方法名及槽/字段类型上下文。第三方异常 message 和栈路径文本各最多4096 UTF-8字节，仅写远端新namespace的 cpu/failure-private.json。CPU-result、stdout及独审输出只保对应hash/bytes、固定常量白名单短文、最多32个函数/文件basename/行号；未知消息不公开。独审核私有文本hash后不转发原文。root如需why可再对该新小私有文件做许可内读取。没有新增raw tensor/JPEG/权重副本。

生产 try-body 在移除 trace 观察调用和 fields 的诊断 stage 参数后与 v1 AST 完全一致，证明见 DIAGNOSTIC_EQUIVALENCE.json；need 仍抛原 ValueError/原message，仅附固定码。v1 原文件未改。此静态核不是原生运行资格。

只复用固定原 v1 namespace 的 data18 文件与 model-metadata-only14 文件，原DATA_MANIFEST SHA5b758107194df4f2599cad3d2598fb4f012eb220c3a75b1052db4d64eac3fcd1。worker重新核SHA/尺寸/完整16输入，metadata/helper/provider原pins不变；receiver不复制视图，launcher只传5个runtime小源（<=256KiB），不再上传JPEG/模型metadata，不读取80答案。运行模块cache/output属于新独占v2 namespace。原模型根/原框架只保既有只读协议，不新driver/训练。

单worker、2CPU排除0/1，client原epoch300覆盖Git准入与传输，child260/supervisor270/清理298；8GiB groupRSS是采样停止阈值不是硬限。新唯一local/remote scope，任何失败或UNKNOWN不重播。无GPU/模型/生成/17项或性能晋级。

CLAIM 的 future CPU_GO 默认false，须root独立审阅后决定一次授权并重封发布。launch_once.py实核publication/MF/claim/own clean，复用原普通SSH入口；当前未连接/未运行。read_once.py仅在原请求known RETURNED后一次消费新诊断小原件（含失败），按同pub MF/SOURCE_PINS校验。无任何当前target执行。
