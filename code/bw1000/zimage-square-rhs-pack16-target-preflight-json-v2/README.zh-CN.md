# Pack16 零卡预检的 JSON 阶段诊断

原 `1103606.2569` 是实际编号的零卡步骤。原子进程返回 1，外层收到了 `JSONDecodeError` 与空 gate，step stdout SHA 为 `3a125244eb06e340c75ab1710ce6053c55f72405e03e29e22d388fc7322b7792`。原日志未标出 JSON 解析位置。原 HOST 在读取保底 `worker.json`（前后两次）、固定清单和成功返回后的容器 stdout 都会解析 JSON。固定清单在解码前已经校验发布 SHA，普通保底生产者以同目录临时文件原子替换 `worker.json`。容器 stdout 混入文本是需要分辨的假设，当前原件尚不能确认根因。

新源分别标记 `standby_before`、`runtime_host`、`runtime_inside`、`container_stdout`、`standby_after`，并保留输入 SHA、字节数和 JSON 解析坐标。将已知的 `/opt/dtk/env.sh` stdout 导向 stderr，退出状态仍有效。HOST 在新独占输出目录内原样保存子进程的 stdout 和 stderr，外层只发布界限内摘要。任何额外或无效 stdout 都会导致明确失败。容器内若已产生结构化错误，则保留其原阶段、子进程退出码和允许公开的诊断摘要。

`render.py --binding <未来原始绑定> --binding-sha256 <SHA> --out <新文件>` 生成单件 `precheck.py`。绑定需要当前 `1103606/f12r2n10` 的不同保底步骤和不同请求、独审 8 张卡正在计算的原证据 SHA、实际 worker SHA、boot SHA 和独立新预检请求。绑定与渲染文件只在已授权环境中保存。新的作用域须另行发布认领并实时核验唯一性、保底状态和资源。实际仍限 1 CPU、256 MiB、0 GRES 和 Slurm 60 秒。`.2480` 和本失败 `.2569` 均不能代用，未准备用于既有作用域的重放命令。当前未访问集群或变更保底。

本机回归执行了真实 Python 子进程和 Git Bash，验证 stdout 被非 JSON 内容污染会被拒绝、stderr chatter 会保留、内层返回 1 时保留其 JSON 阶段，以及旧和不完整绑定被拒绝。尚无新目标容器或 GPU 结果。
