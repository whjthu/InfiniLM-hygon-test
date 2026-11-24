# TRELLIS.2 生成组件的官方权重

`ASSETS.json` 固定微软两个发布仓库的五组配置和权重，合计 7,213,643,479 字节。配置的 Git blob 身份与发行文件字节已交叉核对，权重 SHA-256 来自对应版本的 LFS 对象。下载经已在海光验证的微软 ModelScope 路径和系统 CA。ModelScope 的 `master` 只提供传输，文件准入以固定的微软发行版本、尺寸及完整 SHA-256 为准。

启动器用新的私有阶段目录，最多同时传输两件。每个权重逐段核验 HTTP Content-Range，完整 SHA-256 和 safetensors 数据范围都通过后才建立最终文件。预算 880 秒，由同一次启动和外层超时约束。失败记录保留已验证文件、原 intent 和逐件状态。既有目录禁止重新启动。只使用登录侧下载及磁盘校验，0 Slurm 步骤、0 GRES。任何二进制权重都留在集群存储。

现有 Shape Flow 训练底座不下载。DINOv3 官方模型返回未授权访问，还需已有授权原件或官方授予访问。完整生成另外需要源代码依赖、同更新数的检查点和坐标配准。

[TRELLIS.2 固定发行目录](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/81c44e3aefb363742e68d7eb4540bb65839d9ab3/ckpts) · [TRELLIS-image-large 固定发行目录](https://huggingface.co/microsoft/TRELLIS-image-large/tree/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts) · [DINOv3 官方访问页](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
