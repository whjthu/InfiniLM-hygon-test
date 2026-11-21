# GLM-5.3 fresh K/V 单次编译的 HOST 入口

绑定已发布的 47f8380857d2ec58e549219a09bccafa26211299 与两组 fresh K/V 的精确源包。原请求 b9a7f0b78c566911ed40004fc906ce44 使用 O01 的 1103606/f12r2n13、4 CPU、8192 MiB、0 GRES 和 300 秒。节点身份来自冻结台账，实际准入读两个父作业并要求指定节点无 numbered step。两组 CTA 参数编译为同一个对象。

先用已发布的 fresh-kv-pack-CPU-v1/prepare_target.py 把三件编译源码生成到 HOST 固定目录。项目根下的路径为 incoming/GLM474-fresh-kv-pack-gfx936-CPU-v1-host-source。该目录和源文件由 HOST 入口重验，绑定到容器的只读 incoming 范围。

    /usr/local/python3.12/bin/python3 -I -S -B host_once.py --claim-sha256 <frozen-claim-SHA256>

入口从现有未改的 Deep inverse RoPE controller 复用资源观察、srun 命令、独占目录、单 epoch 和有界终态等待。初始记账在两个父作业中检查同一标签。结果只认原请求下的具体 numbered step。只有 Slurm C0、实际 0 GRES、worker 原 epoch、gfx936 命令、原 M/MD 文件、符号日志和已绑定的 ELF 对象同时闭合，才写 COMPILE_VERIFIED。对象、回执、M、MD 和原终态行均保存 SHA256。

    /usr/local/python3.12/bin/python3 -I -S -B host_once.py --claim-sha256 <frozen-claim-SHA256> --reconcile

只读模式从原 intent 取 epoch 并读取一次实际 accounting。原会话超时、未见终态或存在歧义时保留原状态及结果，不能调用启动模式重提。历史已确认终态或产物如与新观察不一致，入口报错。此模式不改原 controller、编译回执或产物。

本机复用真实 HOST 入口，Slurm、资源观察和目标 worker 为夹具。实际本机 C++ 编译并转为 ELF 的文件只作为产物测试。目标节点、真实编译和 GPU 资格待相应实证。
