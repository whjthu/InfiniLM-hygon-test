# Treasury 三臂通用 caption 开发对照：唯一 dev16 版本

本包只准备base/old_domain/explicit328三臂的通用描述能力开发诊断，不训练新LoRA，不增加Treasury/VGGT之外的路线，不替换原108财政开发题或正式T01质量用例。固定16图。当前真实选中0图，原因是本机只有captions_val2017缓存，正式CIDEr冻结范围尚未明确，故不把其余val图调作开发。本实现严格只接受官方captions_train2017完整member；不把未知ID/缺图片或空评分伪装为就绪。

已本地核官方ZIP尾目录：原档252907541B；目标member的localheader offset159287046、DEFLATE18816055B、解压91865115B、CRC32 3427477015。下一个member offset178103194，故完整localrecord含header/name/extra共18816148B；未来还须实际检查localheader的35字节name和28字节extra。本轮只准备不拉。固定唯一URL为`https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`；它是由[COCO原官方URL](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm)和[AWS官方path-style/CNAME同名bucket规则](https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html)推导，不声称COCO网站直接列出这个新link，也不是运行时fallback。root已有两个0body HEAD原件：原images域证书hostname mismatch/verify62；固定AWS地址正常TLS通过、HTTP200/Content-Length252907541、Accept-Rangesbytes、无Location、ServerAmazonS3及ETag`"f4bbac642086de4f52a3fdda2de5fa2c"`。HEAD不代替实际RangeGET资格，本次child未发新HEAD/GET。

fetch_https只接受上述一个正常CA/hostname校验的HTTPS URL，禁代理/重定向/HTTP降级/整档fallback。必须HTTP206、单一精确Content-Range/Content-Length以及上述单一精确ETag全部符合，才读取body；缺/错/重复ETag也在body读取前拒绝。200body、chunked/gzip及其他最终URL均拒绝。最终按root新增指令实际运行假HTTP检查：保留旧18拒绝分母，再加入缺/错/重复ETag共21拒绝，body读取0、真实HTTP0；仍不能称ETag已有实际RangeGET验证。未重复原大内存或selector测试。根先普通发布再唯一调用execute_once，原始注释只放local-only。

原64图/64 SHA以既有original-bindings SHA5d9e6132…冻结；已有官方val metadata只用于找到这64图的Flickr source group。选择器先固定原64 COCO ID和Flickr photoID排除，再仅按预先固定seed+COCO ID的SHA排序，按图片宽高取8 landscape、4 portrait、4 square，每Flickr组最多1图。排序/配额接口没有caption、答案或模型结果参数。维度配额只是视觉形状覆盖，不能称为语义场景代表性已证明。完整选中16后才按固定ID提取所有实际参考caption；媒体失败、缺caption或评测失败均保原16分母，不回填或按效果重选。

版权分别处理：[COCO官方条款](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)把annotations/website许可为CC BY4.0，但不拥有图片版权，图片须遵从原Flickr权利。本版本只从COCO image license=4且对应[CC BY2.0](https://creativecommons.org/licenses/by/2.0/)的记录选候选；该许可允许商业复制/改编且要求署名，不能把图片统称CC BY4.0。COCO元数据只给原图静态URL，不完整提供作者/title署名信息；未来每图还需原照片页面/作者/许可及credit记录。本轮没有真实16图署名、媒体或许可完成结论。参考caption及原注释不入Git；Git仅维护代码、来源说明、合同和小哈希计数。

未来在R原地或忽略L实际供图后，每张必须执行`image_hash_gate`，真实SHA不得与原64中任何一张相同，且16图自身内容/来源组不重复；当前没有候选媒体，不能宣称零SHA重合。该函数只核JPEG头/字节身份，完整JPEG解码和Qwen原生processor/4096输入仍待下一scope。近重复视觉内容尚未核，不从COCO ID不同推导近重复为零。请求固定为只含图和简短英文描述指令，参考caption独立私有存放，不拼进输入；仍需统一框架原生Qwen plugin/collator实际资格。

评价器只参考COCO作者的[caption evaluation](https://github.com/tylin/coco-caption)及其[CIDEr实现](https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/cider/cider.py)。上游README为旧Python2/Java环境，PTB tokenizer调用Java；当前没有已冻结并实际运行合格的scorer，不能假设现平台可直接用。下一scope必须固定实际源commit/模块SHA、Python/Java/tokenizer JAR及调用、n=4/sigma=6和DF参考范围，以自相似/无关caption等真实fixture确认实际实现后再冻结receipt。本包不实现CIDEr替代、不输出null当分数、不会用样本字符串相等代替效果。

将来三臂必须用同16原图/顺序/答题指令和同generation边界，保留每臂16 missing/cap/invalid等分母；CIDEr仅由合格的实际scorer产生，按同一16参考集合和相同DF规则做开发配对。小样本受IDF与抽样方差影响，结果只作通用能力开发诊断，不是正式/盲测COCO CIDEr、通用训练对照、无预训练污染证明或整体泛化验收。本轮没有scorer运行、模型输出、训练、SSH/GPU/定时任务。

新增完整本地获取/选择scope为一个300秒epoch（含源hash/Git本地dirty检查），父进程不重置期限。必须先普通发布，execute_once对指定publication的MF/CLAIM做git show原字节比较，并拒绝该source路径dirty或untracked；没有git fetch/push/SSH。唯一忽略目录以exclusive mkdir固定，最大一次HTTP open，失败不重播或换目录。WindowsJobObject在放行worker gate之前实际设置并查询2GiB process/job commit硬限制，失败不放行；父进程deadline前5秒杀owned job并留清理预算，inside读循环与解压/选择受同原epoch约束。不是cgroup保证，不使用未验证512MiB假设。

本机一次合成压力运行使用相同parse_decoded_json/select实际源码、91865115B内存JSON、120000图/600000条合成注释和两个18816148B压缩占位buffer，原始合成payload不保存。实际WindowsJobObject峰commit545079296B、采样的OS PeakWorkingSetSize549920768B、父wall2.719秒；解析0.313秒、选择0.890秒，确实超过512MiB。这里只支持2GiB工程预算可行，不是实际官方数据峰内存资格；将来真实运行仍必须同样硬限制且实际计量。整个官方92MB解压只存在有界内存；仅保存18.8MB压缩member，roster/inputs/receipt各≤32KiB、raw selected references≤256KiB。超限或后续缺项保留原16分母和已冻结名单，不新增候选或升级dataset/trainingGO。

最终显式worker2逻辑CPU已接入：从父进程实际允许mask取最低2bit，JobObject配置affinity后逐项查询，再对child Set/GetProcessAffinityMask，全部符合才写gate放行；worker自己在工作前后核mask和count。父controller未加入workerjob，不把它虚构为全进程cgroup保证。一次轻量无HTTP/无大内存烟测实际mask=3/count2，Job及process一致，wall0.110秒。此前2.719秒的大内存探针没有CPU affinity限定，仍保留其内存计量，不把旧耗时冒作新2CPU性能。
