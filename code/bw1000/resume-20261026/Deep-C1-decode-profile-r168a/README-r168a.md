# r168a：保留1187失败、修复前置数据合同

r168的1187在8分29秒真实加载后才触发旧fresh64/accepted64 guard；0个C1捕获，原失败/清理证据保留，不能计作性能实验。实际 prepared64/accepted64/skipped0；真正错误是把9234旧fresh64 gold来源带入9278的另一组64题：旧gold完全不含selected4。CPU525仅查selected4在prepared中，没执行Runtime入口gold集合条件。

新包只修数据前置条件与独立namespace：严格读取现有remote /incoming/Deep-C4-independent64-r9278/dataset64.json（dc1e0d07...）；包内053f9762...仅是LF语义副本，实际预检须parsed全部64题/答案/元数据整体相等，选题、题文、答案、输入token、cap512/context1024/自然EOS均不变。一个纯CPU prepare_pilot函数被CPU_static、bootstrap（Torch/CDLL/Runtime/权重前）和Runtime.run共享调用；核preparedSHA、goldSHA、planned64的accepted/skipped精确分区、selected4未被拒绝、原题文匹配、不截断/pad/不含测试gold、原3shot、tokenizer文件SHA、全部selected input长度/预算、模型EOS/43层几何。没有用删除assert忽略数据不一致。

原T=B=1 owner/inner捕获、OFF7/PROFILE8/OFF9、4自然EOS问题、Native ac937与所有数值/生命周期不变。actual GPU C1支持仍待执行，不以schema通过冒充。CPU预检成功后root另行决定一次有界GPU；本包不自动提交或重播。

本地已验证LF副本换CRLF后的SHA恰为dc1e原始SHA，仅行尾不同；不覆盖remote，也不把053f副本冒充actual原始证据。共享payload回归允许63+1skip只是CPU坏假设回归，当前实际仍64/0不变。
