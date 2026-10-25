# Kimi K3 首个真实在线投机推理闭环（r9046）

32卡 target（TP16×PP2）+1卡 DFlash2 草稿，708/1035 均正常完成，32个 target owner 与1个草稿 owner 正常退出，独立 audit 通过，post KFD 全空。

最高单用户 decode **13.9043 tok/s**，同题 AR **4.87465 tok/s**，约 **2.8524×**；还没有超过15。另一道题10.9638 vs4.90788，约2.234×。普通自然prompt未普遍获益，不能称为所有请求加速。

## 全部实测

|case|AR/spec K|decode tok/s|对同题AR|平均有用输出/轮|proposal接受率|verify s|replay s|draft RPC s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|natural0-K0|0|4.56103|1.000×|1.000|—|13.67644|0.00000|0.00000|
|natural0-K2|2|3.64443|0.799×|1.615|32.05%|9.93718|6.13303|0.89228|
|natural0-K4|4|3.69026|0.809×|1.909|22.73%|9.21127|6.85771|0.75429|
|natural0-K7|7|3.48037|0.763×|1.909|12.99%|9.92120|7.13242|0.76594|
|natural1-K0|0|5.01488|1.000×|1.000|—|12.38623|0.00000|0.00000|
|natural1-K2|2|5.05583|1.008×|2.100|56.67%|7.64963|3.88987|0.68739|
|natural1-K4|4|4.22236|0.842×|2.172|30.17%|8.12637|5.88765|0.66602|
|natural1-K7|7|3.92379|0.782×|2.172|17.24%|8.68912|6.43213|0.67022|
|gsm8k0-K0|0|4.90788|1.000×|0.994|—|31.12125|0.00000|0.00000|
|gsm8k0-K7|7|10.96383|2.234×|5.200|60.48%|8.99208|4.33595|0.65668|
|gsm8k1-K0|0|4.87465|1.000×|0.990|—|20.77579|0.00000|0.00000|
|gsm8k1-K7|7|13.90429|2.852×|6.500|79.46%|4.79864|2.18503|0.34526|

## 口径与质量

13.904294 = 104 个有用 decode token / 7.479703578 秒。排除首个 prefill 产生的 token、EOS、拒绝的草稿与重放输入。decode wall 含 draft、feature capture/传输、target verification、commit/replay 和 emit；排除 prefill。时钟为 native emitter token-ready，尚非 HTTP client。token 会成组输出，P50 ITL 为0不代表没有约0.6秒的长间隔。

GSM8K 官方 test 第0/1题：AR/spec 均分别 parse 出18/3，全部正确并自然 EOS。generated 长度158/106，对应 useful decode156/104。只有2题 smoke，不证明 benchmark 无掉点。两种自然512输入 prompt 都输出至64 token上限，不能算完成任务质量。

## 瓶颈与后续

GSM1 K7 的16轮：verify4.79864s、fullModel replay2.18503s、draftRPC0.34526s（其中draft计算0.31698s），8轮部分接受需要重放。GSM0 replay4.33595s。state-only replay 是首选：缓存必要Conv/KDA状态操作数，仅更新已接受前缀状态，避免重复整模型 MoE/MLA/head。扣除全部 replay 的同轨迹理想值约19.64/15.77 tok/s，仅是零成本反事实，不是已测新性能；实际状态重建仍有成本，接受轨迹也可能变化。

draft RPC 内的计算、H2D和网络等相互包含，不能叠加；跨host capture计时也不能直接相加。原数据保留各host分项与逐cycle。下一轮应加入两轮低收益后自动AR回退，以及同一权重驻留加载的可追加case命令循环。

## 来源与复现

实际 target910375.708、draft910374.1035、targetpost709、draftpost1037；native DSO SHA256 `14bc32c2bede5f70a245bb6c132d94fab49172a761b173978c158581e6c833cd`。草稿第三方lightseekorg/kimi-k3-dflash2，revision e77935fb4804e17eb55085bffd045eae1d779769，span8/7 proposals。原始report与audit远端路径/哈希保存在 terminal compact；源码和CPU验证见 producer index。原始源不再变更。
