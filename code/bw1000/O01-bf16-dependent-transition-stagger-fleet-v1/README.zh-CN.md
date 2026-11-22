# O01 整片分组错峰换段对照

原任务共有 35 台、280 张卡接近同一换段点。五次 640 卡脉冲中，原候选内的非零卡从 592 降至 312。新任务比较同一拨卡上的相同计算、传输和验证，每卡仍为五段，每段 60 秒。每次提交均执行两次 depth9 重放、36 条 BF16 依赖链、一次 761856 B 固定 D2H、两次 host fence，保留原数值与红区校验。实际提交次数与有用行数另记。

每卡先执行原 82/108/130 秒分组启动的实做段。对照沿原三个启动组各自的时序，从第一段开始后第 64、128 秒各执行一段。中间组共同换段的间隔为 [232,236)，另外两组在该区间仍有工作。候选在 [339,399)、[403,463) 的基础上，以完整分片向前 8 秒、不平移或向后 8 秒，优先按节点数均衡。全部工作最迟在原 epoch +471 秒结束。固定顺序为同相后错峰，可能受时间顺序影响。此范围检查调度换段。

发布后由根使用 launch.ship(publication, snapshot, transport, requested=None)。新鲜设备准入继续保护真实训练、推理和原保留。静态排除三个原未决节点、两个外部 KFD 节点及当前 T12 验证节点。上限 74 台、592 张卡，每个 Slurm 步最多四台、每台 12 CPU/64 GiB/8 DCU。保持原协作停机、终态和独立清场规则。

活跃期间在目标已安装的源目录运行 transition_read.py --plan <原 worker-plan.json>，生成两组固定请求及纳秒 epoch。每组调用下列有界采样入口，建议在该组第一点前约 12 秒进入。第一点需在 25 秒内，后续四点间隔 4 秒。

    /usr/local/python3.12/bin/python3 -B "$src/transition_pulse.py" collect \
      --request "$pulse_request" --ledger "$src/ledger.json" --pci-pins "$qualified_full_fleet_pci_pins" \
      --points 5 --max-span-ms 250 --first-pulse-epoch-ns "$first_pulse_epoch_ns" --output-dir "$new_pulse_dir"

两组都需要得到两父 0-GRES 实际终态、80 节点/640 卡/五点，脉冲的跨节点读跨度由同一控制器单调时钟约束。错过的点留在 6400 总采样分母。原工作全部终态并独立清场后，以下命令计算同一选中设备集合的非零卡谷底与忙比例谷底差，并另报全部 640 卡的点值。缺任一阶段或原工作分母时，配对差保持空。点值不能作为持续利用率或三模型验收。

    /usr/local/python3.12/bin/python3 -B "$src/transition_read.py" --plan "$worker_plan" \
      --terminal "$original_reconcile" --aligned "$aligned_outcome" --staggered "$staggered_outcome"
