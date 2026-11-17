# 最终缺段终态只读消费者

准备源码不读取在途RUN。仅根确认原 request `011de10da36149aa9934ded287fdba27` 已终态后执行一次；publication `3973f95839fa84278b756467995c4eb9956e8cc7`，epoch1794907315.2376022/deadline1794907915.2376022，MF144674/CLAIM6d7b。同时核两次旧失败的公版/独审原件 SHA，固定累计prefix17039360字节/bbb745，最终HTTP仅1776788字节。

复用已审消费者的流式成功engine及selection核验：只遍历新完整member一次，同时全SHA、首17MBprefix SHA、末段SHA、精确ZIP header、CRC3427477015及91865115 decoded字节/SHA。每次解压输出最多1MiB，墙钟120秒，不写decoded、不解析完整训练JSON、不重选完整候选池，不读取旧partial。原64排除由已冻结小val image metadata核；新图像内容SHA/近重复仍未测。

成功读四个小私有selection原件，核16/8:4:4/源群唯一/原64 ID与Flickr排除/固定seed已选顺序/完整reference数/无答案输入。失败保真实Job终态/实际suffix字节/幸存文件长度和roster，合格分母保持0/16、幸存未资格roster单列，不解压或重选。无论结果，further_tail_continuation_allowed=false；失败必须STOP_LOCAL_NETWORK_ROUTE，不允许后续续取或全Range重试。passed指独审通过，acquisition_passed另列。

没有HTTP/Git/子进程、图片抓取、scorer或训练调用。原始caption/ID不进Git；仅本地忽略目录存结果。photographer attribution/nativeprocessor/CIDEr均未资格，不提供CIDEr分数或训练GO。未扫描Git历史，不声称全库原样本审计。

根确认终态并review后唯一命令（输出必须不存在、父目录已存在）：

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-treasury-caption-dev16-final-consumer-v1/read_final.py' --output 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/local-only/training-treasury-caption-dev16-v1/final-original-independent-result.json'
```
