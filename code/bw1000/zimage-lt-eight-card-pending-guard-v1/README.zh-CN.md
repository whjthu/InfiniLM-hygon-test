# O01 待确认派发的保底互斥

这份代码只涉及父作业 1103606 的 f12r2n11。根执行器在实际 .training80-dcu-standby.lock 下先写无过期时间的独占 pending 标记，再允许生产绑定前置、GPU 或后置 intent。旧 .2075 不受此门槛影响。标记目录写入中断、原任一步骤命名空间遗留、收据不可读或调度状态未知，都继续保护这个节点。时间流逝与本地命令失败均不解除保护。

    python3 -I -S -B render_bridge.py NEW_GUARDED_BRIDGE.py

根把生成的 bridge 字节和 SHA 用于原 compute bridge envelope 中的 bridge 字段。仍传入原 controller 字节，不能改写此前已启动或发出待确认的 bridge 请求。该新版本只在 compute bridge 调用冻结 controller.run 前安装读取钩子。冻结 controller 在持有实际共享锁后调用 live_inventory，钩子在那一刻排除 f12r2n11。其他候选照既有规则处理。桥中已有异常和启动失败原样保留，o01-pending-decision.json 单独记录排除原因。若只剩受保护节点，原 controller 保持其无可派发节点的结论。

根执行器使用以下接口。results 为 host_scope.RESULTS 的 Path。hold 为同一 host_scope.standby_lock 持有的文件对象。

    reserve_locked(results, hold) -> marker SHA256
    marker_document(results) -> marker SHA256
    authorize_refill_locked(results, hold, host_scope_path, collect, scope, evidence, refill) -> grant SHA256
    finish_refill_locked(results, hold, host_scope_path, collect, evidence) -> release SHA256

reserve 必须先于第一项原步骤动作。根应在每次计划前置或 GPU 时核对同一 marker SHA。共享锁与持久标记还应纳入监测器的候选筛选范围，避免从其他未接入的启动器绕开。源码不调用调度器、传输工具或保底进程。已运行保底步骤照常继续。

原 GPU 实际终态、同 boot 八卡的独立 post 与无持有者均由已发布 HOST 合同核对。授权只放行一个新的普通八卡保底 request，且父作业和八个 PCI 必须一致。根用同一 request 和新鲜设备准入构造独立保底 envelope，仍经带保护钩子的 bridge。只有新保底同 PID 两次八卡实际 GEMM 增长通过原合同，才写 released.json。保底未知时保留 grant；更换 request 不能复播原范围。前置失败或原动作未送达的退役需要独立证据，由根另行处理。

本机验证使用原 controller 源的实际 run 入口与模拟调度、模拟锁接口。目标端共享锁、部署与实时节点状态仍由根核验。无新的 GPU 准入结论。
