# T08 日常室内动作候选：CMU subject79 / trial81

官方动作标注为 brushing teeth，60Hz。发布方采用自定义使用条款：允许研究和作为商业产品的一部分使用，禁止直接转售原数据（含格式转换）；这不是 CC0 或 CC-BY。保留致谢：The data used in this project was obtained from mocap.cs.cmu.edu. 原始动作不进入 Git；不得据此宣称原始数据可再分发或模型发布权利已自动审清。

一手来源：<https://mocap.cs.cmu.edu/>（使用条款和测量限制）、<https://mocap.cs.cmu.edu/subjects.php>（主体）、<https://mocap.cs.cmu.edu/search.php?subjectnumber=79>（逐录制标注）、<https://mocap.cs.cmu.edu/info.php>（ASF/AMC 与单位）。查证日期2026-11-21。实际原发布链接为HTTP，TLS认证=false；不是HTTPS认证过的快照。无发布版本号，以实际原始body SHA固化。未使用镜像、代理、私有认证或爬取全库。

首次v1已正常终态失败：官方页面完整42,202B、ASF已收齐声明7,257B，随后HTTPResponse在读满Content-Length后fp=None触发AttributeError；AMC未请求。v2只请求此前未请求的原AMC，不重取页面或ASF，并逐SHA核原失败和缓存。两次原终态和全部请求分母保留。累计3GET、760,760个成功read字节；1/1骨架、1/1动作、没有补位。原ASF30bones/62channels；940原帧，60Hz，15.65s，不把通道数当HY201。

实际边界：两轮均WindowsJob 2CPU/2GiB commit硬限；RSS另为OS观测。v2 C0/5.156秒，峰进程commit840,040,448B、OS PeakWorkingSetSize采样42,369,024B，不能当形式性能。原数组NPZ以allow_pickle=False实际读取；940×62有限值、原始连续帧号、root米制转换均通过。ASF length=.45，米制因子=.0254/.45；只做根平移单位桥，不计算FK或人体重定向。

整条录制及其未来全部窗/增强共组；subject79仅保留为train候选，不与dev/blind共用。当前train候选1录制，dev0、blind0。发布方说明同一演员可有多个subject号，所以仅subject不重合不证明actor独立。不要把同一940帧的多个128窗计为独立录制。

正式T08保持不变。此前冻结缓存60帧×201、128样本，不满足真实128帧。统一producer需要原motion135、合格body_model生成66keypoints、文本vtxt768/ctxt4096及统计/坐标/重采样身份。当前原CMU Euler/ASF层级到该135结构、身体参数、66keypoints没有合格映射；原producer还是MLU实现。NPZ特意不用motion字段、无201填充、无伪造姿态。手指/拇指是编辑附加而非实测，手/脚趾噪声保留，不宣称测量精度。

下一最短动作：先取得原body_model、135 feature与归一化/坐标定义的可审源码/身份，制定并核验ASF→该骨架的重定向；再用同输入的新A100参考验证真正128×201缓存/Flow Matching首步与恢复。当前无Torch、模型权重、GPU、训练或集群请求，READY=false。不能将这条候选混入原正式A100配方。
