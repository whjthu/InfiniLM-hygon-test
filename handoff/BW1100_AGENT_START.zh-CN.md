# 给 BW1100 Agent 的接入指令

用户指定本仓库作为 BW1000/BW1100 的唯一项目交接仓库。请使用普通 Git clone/fetch/pull/commit/push，无需 GitHub API 权限。

用户已要求两端各自每 10 分钟检查对方更新，实质变化才上传。请读取 [沟通约定](../coordination/COMMUNICATION.zh-CN.md)，在本机启用其中的定时器提示词；BW1000 已设置自己的检查，不代表 BW1100 已启用。首次有实质状态发布时顺带确认启用情况。检查和消费记录只保存在本地 `local-only/`，不上传定时心跳或单纯已读回执。

1. clone `https://github.com/whjthu/InfiniLM-hygon-test.git`，读取 README、AGENTS、coordination/PROTOCOL 和 BW1000 status/results。
2. 建立自己的 `bw1100-experiments` 工作分支。先回传 `status/bw1100/latest.json`：实际硬件/卡数、现用 InfiniLM/InfiniCore/InfiniOps/InfiniRT/InfiniCCL commit、模型权重/量化、工作负载、当前性能、已改源码及其 commit、正在尝试/失败路线。
3. 提交可 cherry-pick 的源码改动或完整补丁；有 dirty tree 必须附 patch SHA。不能仅报一个构建目录或 DSO hash。
4. 先读 tasks，认领与 BW1000 互补的任务；可直接申请跨硬件验证已实现方法。不要把 BW1000 已证伪的同一实现从头再试而不解释差异。
5. 指标以实际用户逐 token 时间线为准，列出 TTFT、decode/user、ITL P95/P99、有效总吞吐/卡、达标吞吐；prefill 单独列。静态 batch、emitter 计时、真实客户端服务须明确区分。
6. 请回传你已经使用的源码与性能证据，首次信息不足写 NOT_MEASURED，不填猜测数据。

BW1000 当前关注：Kimi 词表输出头与低 batch、在线动态调度；GLM 单层 gateup/down 与 attention owner 开销；DeepSeek 单行/多行数值路径诊断。普通 GPUF64 禁用；完整最终 token/EOS 一致是验收条件。具体最新资格以 status/result 时间戳为准。
