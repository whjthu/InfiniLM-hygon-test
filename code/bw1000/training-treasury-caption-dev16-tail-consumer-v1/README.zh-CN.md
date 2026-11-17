# 缺失尾段原终态独审

仅根审阅并确认原请求终结后运行一次；此源码准备不读取在途目录。固定请求 `c4c40fce350442f18365b1c1f96cc05c`、原新 600 秒 epoch、发布 `ea4233ff8f53f6a1df1290d1eff3c6da869f179b`。无网络、Git、子进程、模型或评分器调用，不打开旧 prefix，不修改原件。

成功时对新 complete member 单遍流式读取，同时核全文件及 prefix/tail SHA、精确 ZIP header、原始 DEFLATE CRC/91865115 字节 decoded SHA。解压输出每块最多 1MiB、局部墙钟 120 秒，不落盘或解析完整训练 caption JSON。仅核已有四个小 selection 文件及原64发布 metadata：16图、8/4/4、ID/Flickr 排除、源群唯一、seed/已选顺序、完整 reference 数与 user-only 输入。完整候选池排序不重新运行；完整池选样依据固定发布 producer，不将已选顺序检查冒充独立重选。原有小 val metadata 只用 image 属性作排除。

失败分支固定本次真实 Job124/timeout、suffix read/saved8388608、新 assembled17039360字节；对新 assembled 单遍流式 full SHA，并同时核首8650752字节 prefix SHA、读取前后稳定文件身份，不打开旧 prefix，不解压/重选/补位。保留幸存 roster 与全部16分母；最后缺1776788字节、精确Range176326406..178103193。新的累计 prefix SHA 必须来自本次独审，不预填。`passed=true` 可表示成功审查了失败；必须另看 `acquisition_passed`。

此步骤不下载图像、不提供 photographer attribution、不核新 JPEG 内容重复或近重复、不执行 native processor/CIDEr，不构成数据可评估或训练资格。原始 caption/ID 清单只读写忽略 local-only；reader 不扫描 Git 历史，也不声称完成全库原始载荷审计。Job terminal 依据原 controller 的 child wait/reap，未另做 PID 存活探针。

根确认终态后唯一命令（输出文件须不存在、父目录已存在）：

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-treasury-caption-dev16-tail-consumer-v1/read_terminal.py' --output 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/local-only/training-treasury-caption-dev16-v1/tail-original-independent-result.json'
```
