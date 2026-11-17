# T04 paired training views：仅原位45+45→800+800，准备未执行

同A100本机读取已资格domain45与control45原JSONL及其精确binding/native55/45身份元数据。domain55只过滤45条train记录，不读dev输入/答案，control仅45；无新encode/collator/模型/图像body/权重/网络。两份独立公开资格结果作为本包固定source metadata，原源不改。所有caption和新800行view留R私有，不回PC。

exposure_indices是原unpub adapter函数原文/AST精确复制，seed20261118，800索引由45个索引置乱循环截取，17或18次/源；本包独立，无运行时unpub父依赖，不重旧算法/native测试。每臂800行原45JSONL行字节原样重复，不编辑user、FINAL答案、图片路径或EOS。每条45原source_id SHA与已资格native参考一一核；input_ids/labels/attention/vision字段hash写occurrence ledger，每物理行带original_unique_index0..44和原行SHA。source45身份表只存一份字段参考。实际GPU可能经shuffle/collation/padding/devicecast，故全部CPU batch1哈希仅身份候选，不强行当rank顺序/游标/张量hash相等或训练成功。

R新独占scope含 `domain/`、`control/`，各有train-800.jsonl、exposure-ledger-private.json、occurrence800-private.jsonl、source45-identities-private.json、view-binding-private.json。主exposure-ledger是单个JSON dict，恰含arm(domain|generic)、view_SHA256、original_index_by_physical_row(800项0..44)，control目录的arm=generic，直接供observer config occurrences引用；逐行JSONL仅补充witness。所有完整输入/caption/ledger只R；stdout只paired-result计数/hash，root回传同一小结果和intent/progress即可，禁止取train800/原45/caption正文到PC。最多1600行view和1600occurrence，0真实更新。静态800/(8卡×b1×acc4)=25只是拟预算，actual sampler/rank微批/optimizer步与token曝光由Astra新observer实证。

control原CPU45有效监督token694已逐条计数；domain旧55没记录此数量，必须保持None，不用full_length-firstindex上界冒实测nonignore。weighted800是参考重复后的拟数量，不是训练运行token，更不是两臂token matched。训练过程仍需逐forward真实labelmask/EOS计数、参数更新/clip/scheduler、actualdatasampler/cursor/8rank RNG、checkpoint20→25完整恢复。Astra负责observer，本包不加训练loop/launcher，不修改原396或原CPU包。

CPU2排0/1、单进程无children/GPUFD；root原epoch120包含上传/验证/读写，work100/finalalarm110/client115，所有检查单monotonic剩余缩水。新scope0700，每member独占写；部分失败保两臂45/800全计划和progress/partial，不覆盖旧目录、不重播。root_GO/CPU_GO默认false，root审/pub后唯一普通helper注入INSTALL_PAYLOAD_B64(package+request/publication/sourceMF/claim/epoch)+once.py。模块只stdlib，无nativeprovider/模型导入或外部服务。
