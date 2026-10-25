# r9129 可追踪新batch观察与正式32题准备

跟踪唯一target910380.407/draft910371.528，run batch-r9122b2-222b154b669f9165。Model/draft唯一launcher为nativeowner，不由此目录提交。

CPUobserver已实际910406.253/a14r2n19/PID2420608，c1/mem512MiB，95min有界。每5秒无条件heartbeat（实际StepID/hostname/PID/counter/queueempty_since），每30秒sacct。只保非emitter报告投影及emitter小结，不缓存四份大JSON。实际RPC/结果SLO仍以nativeemitter时钟为准；observer时钟不混成用户clock。当前不是自动质量admitter。

远端 results/Kimi-batch-observer-r9129-batch-r9122b2-222b154b669f9165.json 和 .heartbeat.json。root可调用根目录 read_Kimi_batch_step_r9129.py 经run_stdin_remote_r9046.py读精简snapshot及actualsacct；空读/低版本report保lastknown，并明确fresh/stale，不以gateway/proc做跨host存活证明。

heldout32-native-plan-r9129.json基于原官方固定random32 seed20261026，C2W8(16group×AR/state)与备选C4W4(8group×AR/state)，每group题目真正不同，cap512，question不截断，gold只scoremetadata。全部是pending未入队。先查看初始11cohort真实wholeModel/peruser/quality与I192计数，选最合适一套再入同residentqueue，别两套盲跑。32题不等于C32native并发；C16/32仍需单独pool33/65/T64/4GiB资格。

旧714/1048已COMPLETED0、audit0/normal33/post717+1050空。最终15/32质量覆盖及约25min队列空等事件保存在r9116d/terminal-compact-r9128.json，既不隐藏调度问题也不将15题pilot称完整无掉点。新observer是该观察问题的工程修复，非新性能round。
