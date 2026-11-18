# T08原缓存来源单JSON只读卡

Root已审阅，只有root_GO/RO_GO=true；GPU/rawtensor_GO=false，未执行。实际16055B producer证明frames默认60且会截断原motion，也会对短于args.frames样本zero-pad。不能据旧60frame缓存判断原raw长度，不能单改frames或沿min40造128。

已从原sourceidentity公版ca06c929...字段取得dataset_manifest_SHA256=64b38d39...且matches=true，所以不重新取trainingmanifest f7。只读同dir dataset-manifest.json<=1MiB（producer固定命名），严格核64b38...。返回128行sampleID哈希/原rot6d+joints+text路径及SHA/缓存有效length、原archive和Mean/Std身份；caption只哈希不正文，不返回prompt/不跟随读取sourcepaths、不读NPY/cachetensor/weight、不importproducer/Torch。旧cache metadata长度≤60不是raw全长，输出明确UNKNOWN。

root审/pub后采用既有原T08 A100-26 route仅一次60s/CPU2/AS512MiB/无Slurm/GPU，build_body.py只生成有界stdin，root持久request/bodySHA/attempt后执行；stdout<=128KiB仅落Lignored，不能原样公共发布128sourcepaths。SSH失败或未知不再poll/fallback，当前存在性未证明。

后续先仅用此manifest第一行证明的rot6d/joints两个exactsourcepaths做独立NPYheader-only资格卡（每header<=4KiB、无数组body）核原rot6d.shape=(T,135)、new_joints.shape=(T,22,3)、两者minT>=128及来源SHA；本卡不读取256个raw文件；首对结果不足不扩扫，数量不足必须保留分母，不pad/插值。若可用，则预注册firsteligibletrain顺序、minframes128/frames128的新共同缓存+A100reference，与旧60原件分开。原producer硬torch_mlu/.mlu，必须最小CUDA生产适配且固定同201特征布局，不能直接A100运行或把生成器当训练loop。
