# 自有缓存与真正 HOST-main 的默认关闭候选

仅本地源码，不是GPU-v4、CPU提交包或可执行资源claim。原三GPU失败各108missing、v1/v2失败和旧capacity独审UNKNOWN原样保留。已有producer容量原证为fresh三容器/、/tmp 64MiB，不能据此定位原LLVM writer；NFS statvfs大也不证明userquota。

唯一native入口delta：新增--host-main-preflight，默认False；真实argparse及runpy.run_path(...,__main__)进入main。先核旧dev108/330原39成员、版本和5个ENGINE_PINS，再真实import Torch/Swift ENGINE。HOST分支真实构造108InferRequest、RequestConfig(64greedy)和旧285/新328两AdapterRequest，记录实际__file__/providerSHA；Engine和load_model=True API只做signature.bind，不实例化模型或Engine。真实load_model=False处理器/template仅是独立CPUmetadata。在任何CUDA成功判断和load_model=True前明确return，绝不mock CUDA为成功。移除HOST分支和新flag后默认GPU分支整AST与915601原版相同；未改变输入/64cap/数学。

run_host_main.sh由child shell在任何Python Torch/Swift导入前设置所有cache变量，并将HOME/HF_HOME/HF_MODULES_CACHE显式指向新scope空自有RW目录；不copy旧home/凭据。离线标志开启，token与SSH_AUTH_SOCK只unset不读取。cwd明确从/改为自有RUN，所有模型/input/inventory/adapter/output argv均绝对路径。未来CPU claim必须声明新RUN rw/source ro及--pwd新RUN，与旧挂载差异不能隐藏。

prepare_owned_cache.py默认OFF，只在未来CPU/写入claim批准且显式flag下，于已存在、新鲜、非symlink自有RUN创建cache。计划每10目录最多4KiB canary、共40KiB，写/flush/fsync/读hash后只删除这份刚创建且inode一致的canary，其他文件和目录不清理。仅证明这些字节可写，不代表userquota或任意LLVM/JIT容量足够。本轮完全未执行此代码、未创建cache或canary。

原公开dev108/33039以外的训练53和数据CPU待授权域不访问。未来HOST main的日志仍可含Swift信息：stdout/stderr须完整分别保存，资格读独立host-main/host-main-receipt.json，不能再次把stdout整份当JSON。CPUmetadata不替代GPU数值、质量或性能；所有GO当前false，root另审/publish唯一CPUscope后才可执行，禁止GPU-v4。
