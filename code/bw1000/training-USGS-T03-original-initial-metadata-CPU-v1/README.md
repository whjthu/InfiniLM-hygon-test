# T03 原始初参元数据最短卡

仅准备，尚未远端执行。`read_initial_metadata.py` 仅读原 InternVL 8 rank 的 owner/progress 共16份 JSON；不读 initial.pt、权重、checkpoint、题图或金标。每份≤64KiB、总≤1MiB、stdout≤16KiB，2CPU排除0/1，45秒同原始客户端 epoch。root 发布/采纳后沿既有普通 A100 SSH stdin 入口调用；本包不引入传输或常驻服务。源自身检查 epoch、CPU affinity、稳定文件身份和完整8rank分母；root 客户端准备/发布核验/上传/SSH timeout均须沿同一 origin 缩水，未知不重播。

输出的 initial.pt 大小和全SHA来自原 progress 声明，当前不重哈希 tensor。参数 namedSHA 来自原 owner；与已发布 original200 真正CPU数值独审组合复用，不将此次元数据读取冒充重新安全加载/权重数值资格。各rank归档全SHA可能因RNG/序列化不同；要求的是8rank named参数SHA相等。历史PID仅溯源，不作当前资源准入或终态证明。

`EXECUTION_CARD.json` 列出 T03 配对训练尚缺项。不能把 T04 observer 的硬合同换个case字符串就当 InternVL 合格；保原GC language-only生命周期及原AdamW/cosine。原T03恢复有数值漂移，首个真实域更新耗时UNKNOWN。原17项验收、领域效果、GPU训练GO全部未晋级。
