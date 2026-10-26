# v2 本地 shell source 包（未执行）

仅新增资源配置、common、keeper、GPUinside、唯一launcher及新版本manifest。复用753已审fresh/watch/audit/post，CPU755 provider/native/runner sealed不变；PAIR固定1，INT8单独OFF/ON，T8/T64历史及T64合成低复用三个scenario不变，256整数oracle并入原warm。

GPU_inside_r113v2.sh明确调用run_GPU_r113.run，输入精确GPU-manifest-r113v2b.json，输出Kimi-W2-INT8-register-v2-r113c-${JOB}.${STEP}-node${PROCID}.json。expected-headers-r9125.json已原字节存在。唯一未来入口launch_GPU_r113v2.py必须先读取root发布后另行创建的匹配GO文件；本包不创建GO，也不运行入口。

未来资源保持GPU375 n08/n09，16卡、每节点c32/128000MiB、hard1200/inner1100；CPUkeeper407n20 c1/2GiB。开始前及结束后的fresh/post只作用于这两节点，错误清理只终止本实验实际句柄。

本producer只读取本地既有已审source并写本地文件，做AST/LF与字节哈希检查。没有SSH、远端stage、Slurm、compile、GPU或新CPU门。root按新files_SHA256清单审阅后自行窄stage、普通push并另行GPU GO。旧v1及原v2sealed artifacts未改。
