# InternVL 三 shard / 三固定并发 stream / 单24h（仅源码）

当前全 GO=false，0body/codecfixture/producer/consumer/remote/Git。仅本新包；旧14、362、364、旧partials未读未改，不组装17/26，不建service/tunnel/HTTP/proxy，不换凭据。源码完成不等 runtime/吞吐/模型资格。

范围由冻结原PLAN复制三件：shard02=4,915,962,480B/1dd4d7629deff34807654ca198f45166a532fdbc1bbda3fd80e5e69eebdcdcd4；03=4,915,962,496B/595f3b9043d046ae612f2dbf7cbfcd2465a1c001fcb58e93f78dcaf7cc5976e1；04=2,318,463,688B/1935bfbc7774be5e64adf7ecec5cbefc7c18dae24d498ba4609301c3423b21fd。总12,150,388,664B、11,590块、3文件分母，任何失败保实际已读/已传/完成分母，不按成功子集缩小。

固定接收：1087656/a14r3n04、1087656/a14r3n12、1103606/f13r1n14。源CPU固定[10,11]/[12,13]/[14,15]，fresh affinity必须可用，两两不交，排0..9以避T03 eval2..5/T04 prospective6..9；不足STOP，不替换节点/CPU/topology/方法。源每件2CPU、nice>=10、4GiB AS/0GPU calls。compute每件alloc4/worker2排0/1、4GiB/0GRES，总alloc12/worker6，非12busy。AS不是RSS声明，实际ru_maxrss另报。

relay_once.py 第一条 main 语句建立 T0+86400 原epoch；只核本次普通pub/MF/source，最多6条payload held普通SSH。首receiver2先建立独占common root/modelroot/streams，用tmp写/fsync+原子rename+dirfsync发布CONTEXT.json，含pub/request/epoch/deadline/MF/nonce；3/4只读既有marker，不可补建。三份新control dirs各自独占。源3namespace是existing alignment-assets下3个flat独占目录。

target_host先fresh scontrol parent UID/RUNNING/EndTime覆盖T0+86400/固定节点membership、label dedup和无冲突numbered step，才唯一srun。Slurm秒数在原85020 end内预留30s、向下取整分钟，divmod格式HH:MM:SS；compute READY后另一次exact scontrol step核StartTime/TimeLimit/actualend<=T0+85020，原字段保入READY。若实际schema不支持所需字段则pre-GO STOP，不能凭计划end声明actualend。EndTime/StartTime在Hygon本地时区解析。首receiver READY来自实际compute并已准备空独占partial；其后其他receiver及source启动。每compute先验证共同marker、UID/node、alloc4worker2、aggregate3-file free bytes>12,217,497,528B（statvfs不保证quota）；3个source仅stat原件和资源。只有6份真实READY、三份fresh lease/actualstep、source固定disjoint masks、同不可预测context marker SHA/bytes/UID全通过，各observer FD/path自身稳定且记录host/boot（不比较跨HOST st_dev/inode），才保存release并发送任何GO。前提不足全STOP，不读body。

两个endpoint的stdin bootstrap都为短-c+raw8B精确长度源码包，decoded<=196608B/script<=262144B、Windows commandline<30000字符。不是python-EOF解释器；无sys.stdin.buffer/os.read混用，GO固定3B。PC读source控制和payload始终同一个BufferedReader，按已知长度读取，不吞相邻bytes。

每1MiB raw块独立zlib(level1,DEFLATED,wbits15,memLevel8,默认strategy,Z_FINISH)；128B头冻结magic/version/file/index/offset/rawlen/complen/rawSHA/compressedSHA/reserved24zero。complen<=actualrawlen+64KiB，无rawfallback。源每件一个只读稳定FD，完整disk read/wholeSHA一次，同时blockSHA/compress，最后EOF/wholeSHA/stableFD/path才原件资格。文件wholehash之外的blockhash/compSHA/wireSHA额外输入字节都独立计数。

PC每stream只持有一个frame（临时bytes拷贝计逻辑payload总cap10MiB），chunk<=64KiB；先核头和compressedSHA，再逐头+body rollingwireSHA转发。无解压、base64 payload、大临时文件。HOST只read bounded header/body并转给实际srun，0login payload落盘。目标先核cap和compSHA，decompress(comp,rawlen+1)，严格rawlen/eof/无unused或unconsumed，不无限flush、max_length0或扩大输出。rawblockSHA后stream写/hash唯一新partial，不解码tensor。

每件最后PC必须拿到source全SHA/stableFD正常SSH0，才close target stdin释放EOF commit。compute原wholeSHA/块数/全rawbytes、filefsync/stableFD、pre-rename dirfsync后才rename，再dirfsync+stat。source/PC/receiver整个header+compressedbody rollingwireSHA、wirebytes和块数必须一致。3新shard直接写本新modelroot，但不复制旧metadata/model01，不做Model加载或训练。

producer正常或失败都预接sys.executable新进程read_result.py；独立进程校验MF/本次原小receipts/context/全分母/三端hash与bytes/fsync/stableFD/held正常reap。对每个已知exact新step最多1次普通Hygon freshSSH+sacct（3次总上限），核exact step+label/resource/C0与producer一致，不信producer自报代替fresh。前提缺失或query时间不足保原producer并失败，未确认终态UNKNOWN不重放。reader在本轮未调用，但已在relay真实入口接线，不是TODO。部分独立验证文件只报真实partial denominator，绝不把failed producer升级PASS。

全部层共享原epoch：prepare60/install-admission180/payload84600/stepend85020/host85200/held85500/produceraccounting85800/client85920/freshread86280/held86310/consumer86340/PCcleanup86370/whole86400。worker=min(T0+84600,plannedstepend-180)。fresh每call读<=25/total<=30并随原epoch缩水；reader总caller<=100。所有terminal提前返回，timer取消；不是等24h的服务或poller。首个runtime failure停止组内6个本次heldSSH，失败/未知/缺项保留且无重试。HOST原有step terminal accounting只在原预算内等待，一旦终态即退出。

progress只每>=60s或>=64MiB保存，另有末尾一次；不每块fsync。source read/hash/compress/pipeblocked、PC read/comp/wirehash/writeblocked/progressfsync、HOSTread/srunwrite/progressfsync、compute comp/decode/rawhash/write/fsync分开。并发耗时不加总为网络独占，成功syscall逻辑bytes不冒物理wire/disk bytes。实际压缩率/吞吐/RSS/收益UNKNOWN；最大wire12,911,434,424B不是测量。原900秒控制已逐入口改为新86400常量/PLANbudget，held_child.py保持原exactSHA；无旧控制时限冒充24h。

root独审后才一致采用PLAN/CLAIM的root_GO/provision_GO/READ_GO并重封MF、普通pub，运行入口relay_once.py --publication <40hex> --source-manifest-sha256 <64hex>。本轮仅AST/MF/预算算术，无codecfixture/远端/权重body。旧14和initials保持原路径；三件成功也不证明17/26、native/model/training或质量验收。

共享R的login/compute挂载可能使用不同st_dev。marker只证明各observer见同pub/request/epoch/nonce完整内容、owner及本observer稳定FD/path，不能证明单compute已见全部3个payload。输出各自writer的host/boot/UID、stableFD/path保持严格；后续assembly必须另scope由一个compute observer一次核见3文件metadata/strongpins，本包不做该动作，也不因此宣称同FS/17件ready。

Root independently adopted the exact fixed3-stream scope after two endpoint corrections: rename preserves FD with final-stamp revalidation; originalepoch+85200 hard alarm covers terminal close/save/control. Existing parent leases and step fieldnames were observed read-only, but each new numeric step must still pass its own finite TimeLimit/end and all-six-READY gate before any body GO. Runtime remains unexecuted at adoption.
