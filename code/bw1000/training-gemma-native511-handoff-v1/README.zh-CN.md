# Gemma .511：原生 CPU 收据到后续训练入口的精确交接

本包只解决一个真实接线缺口：原 `.511` 的 producer 收据在 CPU 独占目录，而后续 GPU controller 和 worker 都要求训练 RUN 下的 `native-input-cpu-result.json`，并将其 SHA 写入 intent 的 `native_cpu_receipt_SHA256`。不能拿公开汇总 JSON 重建原 producer、改字段或用“已完成72条”的文字替代它。

## 已真实消费的原件

根提供的本机独审原件 `local-only/training80-user-all-nodes-live-first/gemma-render72-independent-20261122-v1.json` 已实际只读核验：1218B / `e176276ed6e821d1002d41782633b6b40bef56c7587966a036aa42d1f6db6591`。

原 request `8d6d573cc64b4ab3b082b17682043a11`、publication `d62f910ec59c61cf2970308da16a047c33c29b30`、`1103606.511 COMPLETED0:0/36s/f13r1n12/4CPU8GiB0DCU`。独审确认72条、144次encode、72次collator；最大332token、72条合计615 shifted supervised tokens。原 producer 精确344B / `3832af1aef87b88149d955417ad39f665c50fe957a5f91caaf42f44bbc992e80`。

报告未含 source path；路径从已发布 CPU controller 的 RUN 与独审 reader 的 `audit/result.json` 原调用直接闭合：

`/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-render72-native-input-audit-v1/audit/result.json`。

这344B原 bytes **不在本机**，本轮未读取/取回它，只有精确身份与已独审字段。报告的 capture SHA 为 `ff448956bd5f03ba08ff3a072b88c3c83807c1360d2e2c7281ea5cab27b063e6`；未读该72行 capture 原文、问题或标签。

## 入口与最短调用

本机只生成计划：

`python -B handoff.py --independent-report <原1218B报告或其精确副本> --plan-output <不存在的私有计划文件>`。

目标已有普通 HOST 环境，在根核对原 RUN 后安装收据：

`python3 -B handoff.py --independent-report <目标已stage的同一1218B报告> --install-target`。

目标端入口只接受 POSIX；只读上面的固定 source344B，核常规文件/前后路径和FD身份、字节数/SHA、原72/144/72/0model/0update/未观测rankloader/CPU原epoch与390秒范围。之后**原字节**独占写到：

`/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-render72-exposure25-resume20-v1/native-input-cpu-result.json`。

另写 `native-input-cpu-handoff.json`，不重序列化 producer。目标父目录必须预先存在、没有别名；存在收据或原 GPU intent/terminal/step marker 即拒绝，不覆盖或借此复用 UNKNOWN。请先完成此元数据 stage，再创建根后续 GPU intent。本包不查 Slurm、不创建连接/服务，也不读取任何初始化/模型/数据文件。

根后续 intent 应绑定 `native_cpu_receipt_SHA256=3832af1aef87b88149d955417ad39f665c50fe957a5f91caaf42f44bbc992e80`；原 `tools/control_gemma_domain_slurm.py` 和 `tools/run_gemma_domain_pair.py` 对这一文件/字段的检查保持原样。未修改根工具、原 CPU scope、共享 status 或 task。

## 12项新本机检查

实际原1218B报告→精确交接计划已通过。其余是明确的合成 producer/file fixtures：实际 producer344B不在本机，所以 fixture SHA 特别要求与原3832不同，绝不冒原 bytes。

实测报告改字节拒绝、producer model/update/rank-loader/计数/epoch坏值拒绝、文件原字节写入且不重排 JSON、已有收据不重写、已有 GPU intent 阻止 restage；12项通过，Python3.10语法通过。没有重跑 `.511`、processor、真实8卡 sampler、训练或恢复。本机尚未真实交接目标收据。

## 原初始化与8卡 sampler保持独立

原 Gemma LoRA 为476张量/29802496参数，named SHA `09cbdcdb73a8ad1f084b0b4ee2cfd933365064f733caa2a9597aab5ad882938b`；119430531B initial 文件的原自动审核 REJECTED 有明确记录。本包不转移、重提、改名、再生替代或间接绕过它，也不请求新的用户许可。若该原件未依法闭合，训练仍不能开始。

已冻结计划为8卡、每卡batch1、累积4、800 exposure，因此预期每rank100 microbatch、25更新；checkpoint20预期跳过80、剩20 microbatch。`.511` 只做72 unique样本，原 producer 明确 `actual_rank_loader_observed=false`，不能当成这些 sampler/skip 已实测。800 exposure 的有效监督 token 也不能拿615等比例代入（72不整除800且padding/实际取样要记录），本计划保 unknown。

最短剩余链：根在既有目标权限下执行此344B元数据交接→闭原初始化实际授权/身份→确认396共同框架与800view/16模型/端口→fresh8卡PCI/KFD/step准入→实测100/rank、连续25和第20步新进程恢复5→原终态和独立清场/质量。当前新增 READY=0。

没有新数据或许可变化；原 USGS 派生训练数据的来源、train/dev/blind隔离与正式17配方不变。raw producer/capture/数据留私有L/R，Git仅源码和小证据，由根串行发布。
