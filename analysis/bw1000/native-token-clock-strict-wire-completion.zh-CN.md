# 原生 token 时钟的 SSE 完结检查

原 r480 生产端先发送 token 时钟的内部终止记录，再发送标准 SSE `[DONE]`。原客户端根据内部记录的 `emitted` 和已收 token 计数判断完成。若连接在两个边界之间关闭，客户端会给出完成结果。带 `Content-Length` 的响应也可能在已经收到 `[DONE]` 的情况下提前关闭。两种情况均已用本机实际 HTTP 套接字重现。

`tools/strict_native_token_clock.py` 提供与既有 `collect_response(response, start_ns, save_read=None)` 相同的入口。调用进程从固定 r480 包加载 `client_clock.py` 和 `token_clock.py`，工具核对两者 SHA-256。新客户端依次核对同一个请求的 token 时钟、内部终止记录、`[DONE]` 和完整 HTTP 响应体。对于有 `Content-Length` 的响应，响应体结束时的剩余字节也进入判定。内部观察器的覆盖率单列，响应不完整时 `complete=false` 且 `missing_or_rejected=true`。

`save_read` 仍接收原 `read1()` 数据与当次读取的本机单调时钟戳。工具只在接收后辨认 SSE 边界。一次读取返回多个 token 时，沿用真实的相同时间戳。UTF-8 字符或换行跨读取时，按事件实际收齐的时点计时。已有历史计分工具和服务端包按原 SHA 保留。

8 项新回归通过，其中 3 项使用本机实际 HTTP 连接。覆盖完整响应、在内部标记后关闭、收到 `[DONE]` 后 HTTP 字节仍不足、请求身份或结束计数不匹配、观察器溢出和合并读取。另通过 6 项现有服务成本测试。本轮没有重播历史服务记录、运行 Native 模型或获得新的真实逐用户速率。
