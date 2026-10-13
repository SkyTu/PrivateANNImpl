# 测试状态（2026-10-13）

## 连续显存预分配：2 / 10 桶各 300 取最大值

参考用户提供的 EzPC `GPU-MPC/utils/gpu_mem.cu` 预留并复用显存的思路。上游依赖本机不支持的 CUDA 原生 memory pool，因此在 TFHE 无内存池分支加入可选连续 arena：`PRIVATEIMPL_TFHE_ARENA_MB=8192`，每设备一次预留 8 GiB，按 256 字节对齐切分并复用，异步释放使用 event 保护生命周期。默认关闭，底层加密参数与原生比较/条件选择不变。已重新构建 TFHE-rs C API 并通过 ABI 检查。

`bucket_max_once` 使用原有公开 Top-1 归约网络和新 `Evaluator::maximum`（距离 `gt` 加距离/ID 两次 CMUX）。每桶 300 条，完整 23 位距离，距离和 ID 各 FheUint32；每个配置单独运行一次，输入新加密，全部优胜者与独立明文最大值和 ID 关联匹配，含最大 L2 边界和并列值。初始化、加密和解密验证排除在服务器计时外。

| 桶数 × 容量 | 分配方式 | CPU 调度线程 | 服务器耗时 |
|---|---|---:|---:|
| 2 × 300 | 原始 cudaMalloc/cudaFree | 1 | 52.771587 s |
| 2 × 300 | 原始 cudaMalloc/cudaFree | 4 | 39.787205 s |
| 2 × 300 | 连续 arena | 1 | 51.235870 s |
| 2 × 300 | 连续 arena | 4 | **22.725328 s** |
| 10 × 300 | 连续 arena | 4 | **108.347776 s** |
| 10 × 300 | 连续 arena | 10 | 127.846752 s |

本轮两桶的 4 线程预分配版本比原始 4 线程约快 1.75 倍，比预分配单线程约快 2.25 倍。10 桶从 4 加到 10 线程反而慢约 18%，故没有提高默认线程数。这不是统计意义的稳定加速比，也未测原始分配的 10 桶或预分配的 10 桶单线程基线。4 个 CPU 线程通过独立 stream 提交任务，不表示 GPU 只有 4 个线程或 SM。

**进一步定位：预分配尚未消除同步阻塞。** 对 2 × 8、14 个节点的独立小样例跟踪，归约期间 `cudaMalloc/cudaFree` 均为 0，但仍有 **104 次 cudaDeviceSynchronize 和 1196 次 cudaStreamSynchronize**。本地 arena 的同步 `cuda_drop` 分支为保留原有释放安全性而显式等待设备；Rust `CudaVec::drop` 会走该分支，因此当前实现仍存在全设备等待，不能称为完全异步的工作区复用。要进一步减少等待，须先整理缓冲区最后使用的 stream 和生命周期，不能直接删除同步。尚未实现整层批量 PBS。

**同步来源对照已实测确认：** 使用预分配改动之前保存的动态库，对同一个 2 × 8 样例跟踪，得到 **2 次 cudaDeviceSynchronize、1196 次 cudaStreamSynchronize、2708 次 cudaMalloc、2692 次 cudaFree**，结果校验通过。与预分配版本比较，额外的 **102 次全设备同步来自本地新增 arena 的同步回收分支**；两次公共全设备同步来自项目计时器评估前后的等待。1196 次 stream 同步数量相同，来自未改动的 TFHE-rs 路径。原始 `cuda_drop` 使用 `cudaFree`，没有这里新增的显式 `cudaDeviceSynchronize`；不能把新增 102 次归因于上游，也不能认为原始 `cudaFree` 没有同步开销。此次仅审计归属，未直接删除保护缓冲区生命周期的等待。[对照摘要](logs/synchronization_origin_audit.json)、[原分配跟踪](logs/max_original_sync_audit_runtime.csv)。

安装 CUPTI 13.0.85 后，提前初始化和启用采集可以取得内核记录；最终小样例的 7,588 条 kernel 记录，其 start/end **全部为 0**，无法计算 GPU 内核重叠或实际 SM 占用率。主机 API 记录有效；不能从这些结果断言单个 min/max 占满全部 GPU 资源。采集运行含额外开销，未计入上表基线。

报告：`logs/max_2x300_{baseline,arena}_w{1,4}.csv`、`logs/max_10x300_arena_w{4,10}.csv`；[实际分配次数](logs/max_arena_final_launches.csv.allocations.csv)、[API 跟踪](logs/max_arena_final_trace_runtime.csv)、[原始内核记录](logs/max_arena_final_trace_all_kernels.csv)。两桶/十桶预分配四线程的峰值在用切片分别约 717 MiB / 1318 MiB，8 GiB 是预留容量，且不含所有非 arena 分配。100 桶及 1M 大规模组件尚未用新分配器重测。

复现：构建 `bucket_max_once` 后运行 `PRIVATEIMPL_TFHE_ARENA_MB=8192 ./build/privateimpl/bucket_max_once 2 300 4 logs/my_max_2x300.csv`；十桶把 `2` 改为 `10`，省略环境变量使用原始分配路径。`test/` 保持四个文件。

## SM 与 stream 核对：硬件可以重叠，当前算子仍是逐节点调用

针对“是否没有给多个桶分配多个 SM”的检查，新增两个独立诊断，均已构建并运行。没有修改生产比较、选择或网络调度，没有重跑大型基准。

1. **硬件具备跨 SM 并发能力。** `cuda_overlap_probe` 在两个非默认 stream 各提交一个 8-block、每 block 32-thread 的小内核，每个 block 持续约 5 ms。设备端记录 `%globaltimer` 和 `%smid`：普通内核跨两个 grid 的最大 block 时间重叠为 **4.995070 ms**，cooperative 内核为 **4.996100 ms**，两种情况均观察到合计 **16 个不同 SM**。设备报告 142 SM、支持 concurrent kernels。这只证明这张 vGPU 能重叠执行这些小内核，不能代表 TFHE 的实际重叠率或占用率。
2. **没有发现所有工作线程误用同一个默认 stream。** 本地 TFHE-rs 的 CUDA server key 克隆共享底层 key，但为线程创建独立 `CudaStreams`；底层用 `cudaStreamNonBlocking`。只读 launch 拦截也未记录到空 stream，多个 launch 形状各观察到 4 个 stream。
3. **真实 TFHE 运算会启动多 block，但调用粒度很细。** 只运行 2 桶各 4 条、共 6 个 minimum 节点，两个距离/ID 优胜者均正确。归约期间记录到 **1,686 次普通 launch + 1,566 次 cooperative launch = 3,252 次**，即本次平均 **542 次 launch/minimum 节点**。普通 grid 为 1–1,856 blocks、128–1,024 threads/block；cooperative grid 为 2–64 blocks、256 threads/block、32 KiB 动态共享内存/block。这些是启动参数，不是实际 SM 占用测量，也不表示每个节点都是单一内核。

当前 `minimum` 用原生距离 `gt` 加距离/ID 两次 `if_then_else`；同层可调度多个这种高层调用，但还没有把同层所有桶的比较组织成底层批量计算。结合上一节跟踪到的大量工作区分配和释放，下一步应检查批量比较/选择、工作区复用与同步边界，而不能仅通过增加主机线程或声称“GPU 已经满载”解释性能。普通 CUDA launch 由调度器把 blocks 放到可用 SM 上，当前 TFHE 接口不提供“每桶指定几个 SM”的参数；参见 [NVIDIA kernel launch 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。

**仍未确定的部分：** launch 拦截不采集 GPU 执行时间线，不能量化真实 TFHE 内核的重叠，也不能区分分配、同步、计算各自造成的墙钟开销。小规模拦截运行服务器时间为 **441.568 ms**，含诊断开销，不作为无拦截性能基线。本轮完成定位检查，尚未实现或验证性能优化。

结果：[诊断摘要](logs/tfhe_launch_diagnosis.json)、[硬件时间戳](logs/cuda_overlap_probe.csv)、[TFHE launch 形状](logs/tfhe_launch_shapes.csv)、[小规模正确性及计时](logs/bucket_launch_check.csv)。复现命令见 [实验说明](experiment/tfhe_rs/README.md)。

## 并行说法修正与 CUDA API 跟踪

当前已证明的是 **4 个主机算子调用可以重叠**，没有证明 GPU 内核有效重叠，更没有证明桶级吞吐获得明显加速。单桶 20.080 秒、两桶 38.196 秒和 100 桶 2112.137 秒表现为接近线性的工作量增长；此前将 `peak_inflight` 称为 GPU 并行证据不成立。接口注释已明确这一限制，小规模连续实验的日志标签改为 `HOST_CONCURRENCY`。

对两个 300 条桶只做一次 Nsight 跟踪，`cudaProfilerStart/Stop` 将采集范围限定到服务器归约，不含密钥初始化、输入加密及解密校验。598 个 minimum 节点，记录到 4 个主机线程：

| 已采集的 CUDA API | 次数 | 各线程 API 时长之和 |
|---|---:|---:|
| `cudaMalloc` | 115,420 | 44.559 s |
| `cudaFree` | 115,416 | 44.896 s |
| `cudaStreamCreateWithFlags` | 1,200 | 0.354 s |
| `cudaStreamSynchronize` | 50,836 | 0.413 s |

分配与释放合计约占记录到的 API 线程时长的 **59.8%**。这些时长跨线程重叠，**不是墙钟占比，也不是分配本身的纯 CPU 计算时间**，可能包含对 GPU/驱动的等待。设备不支持异步内存池，当前后端使用 `cudaMalloc/cudaFree`；原生比较和选择中反复创建/释放工作区是具体可检查的优化位置。尚不能定量分离分配、GPU 计算及同步之间的因果贡献。

**GPU 内核重叠率仍未知。** 本机 Nsight Systems 2024.6.2 对 CUDA driver 13.0 报版本不兼容，并提示部分 CUDA 事件可能缺失；导出的 SQLite 没有 GPU kernel activity 表，只有主机 CUDA API 和同步记录。因此不能据此断言全部 GPU 内核严格串行，也不能声称已定位唯一根因。

跟踪运行归约 **41.912754 s**、两个优胜者校验通过；有采集开销，普通基线仍为 **38.195831 s**。新增实验环境开关只控制 profiler 采集范围，没有更改密码运算或生产调度。报告：[摘要 JSON](logs/bucket_two_300_profile_summary.json)、[Nsight 文件](logs/bucket_two_300_profile.nsys-rep)、[SQLite](logs/bucket_two_300_profile.sqlite)、[跟踪运行 CSV](logs/bucket_two_300_profile.csv)。

## 两个桶各 300 取 1：按要求单次补测

运行 `./build/privateimpl/bucket_parallel_once --two-buckets`，只执行这一个配置，未重复单桶或 100 桶测试。同一 L40S-48Q，两个桶在一次 `bucket_minima()` 调用内按层合并调度，共享 4 个工作线程；598 个 minimum 节点，9 层，距离/ID 各 FheUint32。数据种子与上一轮相同，第一桶复用相同的 300 个明文距离，第二桶为随后生成的 300 条；所有输入重新加密。

- **两桶服务器归约：38195.831 ms（38.196 秒）**，两个桶的最小距离和关联 ID 均通过验证，退出码 0。
- `workers_used=4, peak_inflight=4`，统计的是重叠的主机算子调用。
- 上次单桶为 **20.080361 秒**；这次两桶约为其 **1.902 倍**。两倍单桶时间为 **40.160722 秒**，本次比这个参考少约 **4.89%**。这是不同运行的单次观测，不能将差异直接归因于并行收益。
- 结果支持用户的质疑：当前实现的耗时仍近似随桶数增长，没有体现明显的桶级吞吐提升。网络可并行、主机调用重叠与实际吞吐加速是不同的测量。
- 初始化 **0.774844 秒**、600 条输入加密 **2.200116 秒**、客户端校验 **0.000588 秒**，均排除在归约时间之外；实验墙钟 **41.428922 秒**。

结果：[CSV](logs/bucket_two_300_once.csv)、[日志](logs/bucket_two_300_once.log)、[进程计时](logs/bucket_two_300_once.time)。测试入口新增 `--two-buckets` 模式，默认旧配置及其历史结果保留，`test/` 仍为四个文件。

## 桶并行核对：单桶 300 取 1，以及 4 / 16 个调度线程

针对“300 取 1 应比 500 取 15 快，桶之间可以并行”的核对，新增小型入口 `experiment/tfhe_rs/bucket_parallel_once.cpp`，复用同一个生产 `bucket_minima()` 和层执行器。每个配置仅运行一次，全部通过；没有重跑 30,000 候选的大型归约。

| 公开工作负载 | minimum 节点 | 层数 | 工作线程 / 峰值在途调用 | 单次服务器时间 |
|---|---:|---:|---:|---:|
| **1 桶 × 300 条，取 1** | 299 | 9 | 4 / 4 | **20.080361 s** |
| 100 桶 × 4 条，每桶取 1 | 300 | 2 | 4 / 4 | 20.641921 s |
| 同一 100 × 4 明文输入及网络 | 300 | 2 | 16 / 16 | 21.818049 s |

单桶 300 取 1 确实远快于上次 500 取 15 的 326.342006 s。此前 2112.136709 s 是 **100 个容量 300 的桶合计**，总工作量 29,900 个节点；绝不是单桶耗时。

后两项固定输入明文、编码和网络，各自重新加密以避免输入驻留状态复用。实测确有 16 个重叠的主机算子调用，但这一轮比 4 线程耗时多约 **5.7%**，增加线程未带来收益。因此生产默认仍为 4；这只是单次诊断，未测定稳定的加速比，也不能据此预测所有规模的最优线程数。

所有桶同层的节点本来就合并执行，共享一张 L40S-48Q 的算力和显存带宽；并行依赖深度不等于固定 GPU 上的完成时间。每个 TFHE 算子内部已有 CUDA 并行。当前 vGPU 报告 `memory pools supported=no`，本地后端回退到 `cudaMalloc/cudaFree`，还没有将整层打包为一个批量算子。本轮当时未采集 profiler 数据，不能将瓶颈定量归因于启动、分配、传输或 PBS 中的任一项；后续诊断见本文顶部。

距离来自固定种子的 128D uint8 L2，完整 23 位距离与 20 位 ID 分别存 FheUint32。1 + 100 + 100 个桶优胜者全部与独立明文最小值及原始 ID 匹配。三项评估合计 62.540331 s，实验墙钟 67.660497 s，密钥、加密、验证分开计时。复现：`cmake -S . -B build/privateimpl`，`cmake --build build/privateimpl --target bucket_parallel_once -j 4`，`./build/privateimpl/bucket_parallel_once`。

结果：[CSV](logs/bucket_parallel_once.csv)、[日志](logs/bucket_parallel_once.log)、[进程计时](logs/bucket_parallel_once.time)。`test/` 仍保留四个文件。

## 1M / C=500 / u=15 / B=100 / k=10 单次组件测量

实验入口为 `experiment/tfhe_rs/cluster_bucket_once.cpp`，复用现有 Server 算子。L40S-48Q 单次运行 **PASS，退出码 0**，全部 1M BFV 距离（含补齐位置）、500 个转换结果、Top-15、100 个桶优胜者和最终 Top-10 均通过明文参考校验。结果：[CSV](logs/cluster_bucket_once.csv)、[日志](logs/cluster_bucket_once.log)、[进程计时](logs/cluster_bucket_once.time)。

- 128D uint8，500 个合成均衡簇，每簇 2,000 条；30,000 个候选分到 100 个容量 300 的桶。完整 23 位距离和 20 位 ID 分别存为 FheUint32，ID 不参与排序比较。
- 100 桶归约总计 **29,900** 个节点、9 层；每个节点调用一次距离 `gt` 和两次字段 `if_then_else`。归约时间覆盖所有 100 桶，不能称作单桶 300 选 1 的时间。默认 4 个主机工作线程，每个算子内部使用 CUDA 并行计算；未实现将整个网络层融合为批量 GPU 算子。
- 质心 Top-15 为 3,008 个完整 minmax 节点及 496 个 minimum 节点，共 35 层；最后 100 选 10 为 633 个完整节点及 96 个 minimum 节点，共 25 层。

| 实际执行的服务器组件 | 单次完成时间 |
|---|---:|
| 1M BFV 距离，245 批 | 3.006238 s |
| 500 个 BFV 质心距离 | 0.012500 s |
| 500 个质心 BFV→TFHE 转换，完整精度 | 74.493564 s |
| 500 个质心选 Top-15 | 326.342006 s |
| 30,000 条公开位置随机分桶 | 0.002820 s |
| 100 桶各 300 选 1，合计 29,900 个 minimum 节点 | **2112.136709 s（35 分 12 秒）** |
| 100 个桶优胜者选 Top-10 | 65.757756 s |
| **已测服务器组件合计** | **2581.751594 s（43 分 2 秒）** |

合计由未舍入的计时值计算，可能与逐项舍入后的相加略有差异。桶内归约和最终 Top-k 都观察到 `workers=4, peak_inflight=4`，后者仅表示重叠的主机算子调用。桶归约约占已测组件时间的 81.8%；按总工作量归一化约 **70.64 ms/节点**，按 100 桶均摊 **21.121 s/桶**。这两个比值不是单独测得的算子延迟或单桶延迟，也不能用于推算无限并行时的速度。

客户端密钥和服务器初始化 **8.575570 s**，客户端查询与候选输入加密合计 **108.938145 s**，解密验证 **0.181021 s**；另有 BFV 列准备 **0.128352 s**。实验函数墙钟 **2700.979498 s**，外部进程墙钟 **45:01.40**，包含输入生成、初始化、客户端工作和对象清理。主机最大 RSS **18,094,892 KiB（约 17.26 GiB）**；没有使用 profiler 测量 GPU 峰值或内核重叠率。

相对于公开合成的 30,000 候选集合，允许截止距离并列的 **distance recall@10=0.9**。每桶只留一个最小值导致一个近邻被同桶更近的记录淘汰；桶优胜者和最终 Top-10 的计算本身正确。这不是实际数据集的全局 ANN 召回率，也不构成秘密候选路由的验证。最终 ID 为 `[760481, 760885, 761658, 761937, 488622, 761869, 488253, 488932, 760271, 489879]`。

**范围限制：这不是新的完整秘密查询流水线。** 原来已搭建的连续链路先转换全部距离，再做 TFHE gather。新的质心先转换顺序还缺少 TFHE 选择结果控制的 BFV oblivious gather。此实验的 30,000 个候选由公开合成数据的明文参考预先准备，再由客户端直接加密成 TFHE；没有用解密后的簇 ID 进行服务器路由。候选转换和该 gather 均未执行，组件时间之和不包含它们。候选客户端加密 **108913.889 ms**，单独统计。

复现：`cmake -S . -B build/privateimpl`，`cmake --build build/privateimpl --target cluster_bucket_once -j 4`，`./build/privateimpl/cluster_bucket_once`。仅运行一次，不重复采样；`test/` 保持原有四个文件。

## 最新单次 BFV 1M 平方 L2 实测

按用户要求单独运行现有 `bfv_distance_test` 一次。L40S-48Q，1 个加密查询对 1,000,000 条明文候选，128 维、uint8 元素；BFV N=4096、t=8380417，245 批，末批 576 条真实记录。

- **百万条距离服务器评估：3085.670 ms（3.086 秒）**，吞吐 324,078.709 条/秒。
- 百万工作负载的查询加密一次：23.482 ms，排除在服务器计时外。
- 测试程序墙钟：4465.721 ms，包含前置 7 个小型回归、密钥初始化、数据生成、加解密和全部校验，不能视作纯百万条距离时间。
- 全部 1M 个距离、末批补齐位置以及查询未被修改的检查通过。

使用 `Enc(||q||²) - 2Σ Enc(q[d])·x[d] + ||x||²`，每批 128 次点积项明密文乘法与 1 次查询范数广播乘法，总计 31,605 次 `multiply_plain` 调用。计时涵盖完整距离方法中的验证、明文编码、数据传输、GPU 乘加与同步，未独立测量乘法内核。未执行 TFHE 转换或 Top-k。

复现：`./build/privateimpl/bfv_distance_test`。结果：[原始日志](logs/bfv_l2_1m_once_20261012.log)、[结构化报告](logs/bfv_l2_1m_once_20261012.json)。这是一次运行，无重复统计；使用初始化后的上下文执行百万工作负载。


## BFV 密钥大小：与 1M 距离测试相同参数

使用 `Client` 的默认 BFV 上下文（N=4096，t=8380417），生成同参数的新密钥对。密钥使用 3 个 RNS 模数，每系数存为 8 字节；密文使用 2 个 RNS 模数。调用库的 `save()` 计数，密钥内容只经过丢弃数据的计数流，不输出或写入密钥文件。

| 密钥 | 原始载荷 | 库序列化字节数 |
|---|---:|---:|
| 私钥 | 96 KiB | 98,328 |
| 公钥 | 192 KiB | 196,624 |
| 合计 | 288 KiB | 294,952 |
| 重线性化 / 旋转密钥 | 0（不生成） | 0 |

当前明密文线性距离计算复用一套密钥，大小不随 1M 记录或 245 批增加。公钥用于客户端加密查询，私钥由客户端保留；当前 Server 的评估密钥包不携带 BFV 公钥，距离运算无需额外 BFV 评估密钥。上述不包含上下文预计算、CUDA 内存池、查询/结果密文或 BFV→TFHE 转换密钥。

复现：`cmake --build build/privateimpl --target bfv_key_size -j 4`，然后 `./build/privateimpl/bfv_key_size`。源码：[key_size.cpp](experiment/bfv/key_size.cpp)，测量结果：[JSON](logs/bfv_key_size.json)。

## BFV→TFHE 转换密钥的大小与 seeded 压缩

使用与 `bfv_tfhe.rs` 相同的维度、噪声分布和分解参数，在独立探针生成一套新密钥，统计真实数组与序列化字节数。探针完成 seeded 密钥生成、解压、Fourier 转换和 GPU 上传；没有重复运行距离或密文转换基准，也没有验证这套新密钥的完整转换正确性。生产转换适配器未修改，seeded 保存/加载尚未接入。

| 专用转换材料 | 展开原始载荷 | Seeded 原始载荷 |
|---|---:|---:|
| BFV 4096 → TFHE 2048 维密钥切换 | 448.21875 MiB | 224 KiB |
| TFHE 2048 → 辅助 769 维密钥切换 | 24.0625 MiB | 32 KiB |
| 高精度 bit-extraction 自举密钥 | 576.75 MiB | 288.375 MiB |
| 输出 bit 自举密钥 | 192.25 MiB | 96.125 MiB |
| **合计** | **1241.28125 MiB（1.21219 GiB）** | **384.75 MiB** |

Seeded 四把密钥的实际序列化总和为 **403,439,920 字节**，比当前展开载荷减少约 **69.0%**。另需一份普通 TFHE-rs GPU 整数评估密钥，当前本来就使用 `CompressedServerKey`：其序列化为 **120,668,379 字节（115.078 MiB）**。一起保存/传输约 **524,108,299 字节（499.829 MiB）**，不计外层协议封装。密钥在所有记录和批次间复用，大小不随 1M 条记录增加。

Seeded 格式在密钥生成时保留随机种子和密文 body；加载时重建随机 mask，然后转换/上传为 GPU 运算格式。节省存储和传输量，GPU 运算密钥仍需展开。当前生成的普通展开密钥没有保留这种格式，需在客户端改用 seeded 生成接口并实现序列化/加载，不能直接对现有 Fourier 数组调用 seeded 压缩。

本次四把展开密钥的 GPU 可用内存下降 **1,310,130,176 字节**，普通 TFHE GPU key 另下降 **317,128,704 字节**；合计约 **1.5155 GiB**。这些是 `cudaMemGetInfo` 观测增量，可能包含分配器/库初始化开销，不含转换中间密文和计算工作区。生产转换对象目前另外保留约 **1.21219 GiB** 的 CPU KSK/Fourier 载荷；完整流水线若又创建独立 Top-k evaluator，还会有额外 GPU key，不能用探针值代表整个进程峰值。

复现：`bash experiment/bfv_conversion/check_key_size.sh`。源码：[key_size.rs](experiment/bfv_conversion/key_size.rs)，结果：[JSON](logs/bfv_tfhe_key_size.json)、[日志](logs/bfv_tfhe_key_size.log)。上游机制见 [TFHE-rs 密钥压缩说明](https://docs.zama.org/tfhe-rs/fhe-computation/data-handling/compress)。

## 当前：距离与 ID 分离，取消 ID 并列排序

记录已改为两个独立原生密文：23 位距离使用 FheUint32，ID 使用 FheUint32。完整 uint32 距离的兼容路径用 FheUint64 容纳无效哨兵，ID 仍为 FheUint32。旧 RankLayout/FheUint128 接口已移除。

- `minimum`：只对距离调用一次 `gt`，同一个加密条件选择距离和 ID，两次字段 CMUX。
- `minmax`：一次距离 `gt`，四次字段 CMUX。没有 ID 排序比较，并列可任选。
- gather 仍对加密簇 ID 调用相等比较来路由；有效性通过距离哨兵检查。比较条件始终为密文，访问顺序不受秘密选择影响。
- BFV 转换输出改为独立距离和 ID 的双字段包；适配库已重建，使用 `bfv_tfhe_convert_split` 防止误用旧包格式。
- 测试参考允许任意并列项，同时验证距离顺序、原始 ID 对应关系及输入记录的出现次数。`test/` 保持四个文件。

字段分离回归仅做小规模正确性检查，每个配置执行一次；当时未重复百万距离、500 个转换或 100 条连续流水线的旧基准。随后按用户要求单独重测的百万 BFV 距离见上节。

| 本轮检查 | 结果 | 服务器评估时间 |
|---|---|---:|
| CPU 网络、布局、并列参考、参数边界 | PASS | 不涉及 GPU |
| TFHE Top-k/gather/桶内最小值，33 次不同组件调用 | PASS | 32878.051 ms 合计 |
| 11 个完整精度 BFV→TFHE 边界及转换后的 Top-k/gather | PASS | 转换 2723.127 ms；组件合计 6678.561 ms |
| 11 个截低 7 位 BFV→TFHE 边界及转换后的 Top-k/gather | PASS | 转换 2810.345 ms；组件合计 7147.442 ms |

这些是不同检查的耗时，不是百万查询延迟。完整精度与截断模式都验证了双字段输出；截断产生的并列使用新的任意选择规则。GPU 组件还直接验证 `minimum((4,3),(4,1))` 保留左记录，ID 不再决定并列顺序。桶间并发诊断为 workers=3、peak_inflight=3。

日志：[CPU](logs/split_fields_cpu_checks.log)、[GPU 选择组件](logs/split_fields_tfhe_component.log)、[完整精度转换](logs/split_fields_conversion_full.log)、[截断转换](logs/split_fields_conversion_drop7.log)、[构建](logs/split_fields_build.log)。

### 1M 容量：N=1,000,000，C=500，u=100，k=10

实测一个距离密文和一个 ID 密文，各有 16 个 radix 块，每块 2049 个 u64：每条原始载荷 **524,544 字节**，两字段序列化大小合计 **526,896 字节**。按公开数量乘算：

| 内容 | 原始载荷下界 |
|---|---:|
| 1M 条输入 | **524.544 GB** |
| 200k 条 gather 输出 | **104.9088 GB** |
| 两者同时保留 | **629.4528 GB** |
| 500 个质心，另计 | 0.262272 GB |

不含密钥、对象元数据、计算工作区和后续中间结果；GB 使用十进制。下界约为旧 FheUint128 表示的一半，仍超出测量时约 242.9 GB 的可用内存。容量探针返回 77 / CAPACITY_BLOCKED，**没有执行百万 gather**，运行时间为 N/A。数据：[容量 JSON](logs/tfhe_gather_1m_u100_split_capacity.json)。压缩或固定公开顺序的流式处理尚未接入当前接口。

---

## 历史 packed rank 版本结果

以下各节保留字段分离前的实际日志与时间，含当时的 ID 并列规则和密文表示；不代表当前双字段版本的性能。当前连续实验默认报告路径为 `logs/selection_pipeline_split_once.csv`，旧报告未覆盖。


## 单次连续链路：BFV 距离到桶优胜者 Top-k

`selection_pipeline_once` 在 L40S-48Q 上完整执行一轮并通过，退出码 0。复用现有距离、转换、Top-u、gather、桶内最小值与 Top-k；本轮补上 `Server::random_buckets()` 的公开随机位置路由。服务器阶段之间没有解密，结束后客户端统一核对所有阶段和最终结果。

| 参数 | 本次配置 |
|---|---|
| 数据 | 100 条，128 维，uint8（0–255） |
| 聚类布局 | 10 个合成均衡簇，每簇 10 条；质心为簇内逐维均值四舍五入 |
| 选择 | u=4，gather 输出 40 条 |
| 分桶及输出 | B=16，每桶容量 3，共 8 个公开补齐槽，k=10 |
| 距离与密文 | 完整 23 位距离、20 位 ID、无截断，FheUint128 |
| BFV | 环维度 4096，明文模数 8380417；一个 BFV 批次包含 110 个待转换距离 |
| 种子 | 数据 20261012，分桶 20261013 |
| 并发 | 桶内最小值和优胜者 Top-k 均为 4 个工作线程，peak_inflight=4 |

| 连续服务器阶段 | 单次耗时 |
|---|---:|
| BFV 距离（100 条记录 + 10 个质心） | **13.422 ms** |
| 110 个真实 BFV→TFHE 转换 | **22327.483 ms** |
| 10 个质心 Top-4 | **3940.948 ms** |
| TFHE gather，输出 40 条 | **17194.208 ms** |
| 公开位置随机分桶 | **0.403 ms** |
| 16 桶并行最小值，32 次 min、2 层 | **2760.380 ms** |
| 16 个桶优胜者 Top-10 | **8295.750 ms** |
| **服务器阶段合计** | **54532.592 ms** |
| **连续服务器墙钟（含阶段间调度）** | **54532.863 ms** |

密钥/服务器初始化 **8581.416 ms**，查询加密 **25.031 ms**，客户端解密验证 **78.633 ms**；整个实验墙钟 **63521.992 ms**。各阶段前后同步 GPU，统计完成时间。公开分桶只重排密文句柄，在 CPU 执行；BFV 距离和 TFHE 比较/选择调用 GPU，转换仍包含 CPU 的 RNS/LWE 提取。

110 个转换结果、40 个 gather 结果、16 个桶的布局与优胜者、最终 10 条记录均与独立整数参考完全一致。客户端最终 ID 为 `[588087, 580168, 627682, 635601, 603925, 596006, 564330, 619763, 366355, 334679]`。相对全部 100 条记录的精确 Top-10，单种子 **recall@10=0.8**；该数字反映质心选择/每桶只留一条的近似规则，不能当作真实数据集召回率。并发计数证明 API 调用重叠，不是 GPU 内核重叠率。

复现：`./build/privateimpl/selection_pipeline_once`；只执行一轮，无预热和重复采样。源码：[selection_pipeline_once.cpp](experiment/tfhe_rs/selection_pipeline_once.cpp)，结果：[CSV](logs/selection_pipeline_once.csv)、[日志](logs/selection_pipeline_once.log)、[构建](logs/selection_pipeline_build.log)。`test/` 仍保留四个文件，本轮未重跑整套组件回归。

范围截至加密桶优胜者 Top-k，尚不含输出净化、仅 ID 响应封装、网络通信及真实数据集聚类流程；百万规模仍受下述内存限制。不能报告完整生产 ANN 或百万规模端到端通过。

## 上一阶段：并行桶内最小值

已实现 `Evaluator::bucket_minima()` 和 `Server::bucket_minima()`，位于现有 `tfhe_select` 模块。复用 `topk_schedule(capacity, 1)` 与 `execute_layers()`，合并各桶同层的原生 GPU `min`。公开等长桶、加密无效槽位、固定 B 条输出；不按秘密占用跳过桶。当时尚未接入随机路由；随后完成的连续实验见上节。

本轮 L40S-48Q 单次 GPU 组件检查通过：

| 检查 | 服务器评估时间 |
|---|---:|
| FheUint32，4 桶 × 5 槽，16 次 min、3 层 | **650.356 ms** |
| 上述 4 个桶优胜者 Top-3 | **299.331 ms** |
| 相同 4 × 5 布局，全部无效 | **587.747 ms** |
| FheUint128，3 桶 × 2 槽，3 次 min、1 层 | **239.676 ms** |
| 每桶容量为 1，无比较 | **0.003 ms** |

第一个样例包含空桶、单条桶、并列距离、满桶与非 2 次幂容量，桶内最小值加后续 Top-3 共 **949.687 ms**。宽整数样例验证最大 uint32 距离/ID 与 bit-64 无效哨兵，同时记录 `workers=3, peak_inflight=3`；每桶只有一个 gate，因此确认不同桶的原生 API 调用发生重叠。该计数不代表 GPU 内核重叠程度或相对串行加速比。

测试还验证固定输出数、32/128 位、非法形状/混合布局拒绝和输入不变。整套现有 Top-k/gather 加新增桶测试共 **32 次评估，21776.445 ms**，总墙钟 **23162.600 ms**；这不是一次桶归约的延迟。原始数据：[组件日志](logs/tfhe_bucket_component.log)、[构建日志](logs/tfhe_bucket_build.log)。本轮没有重跑百万 BFV 距离、转换或 100 选 10 性能基准，`test/` 仍为四个文件。

性能限制：当前每个 gate 单独调用 TFHE-rs；多线程向同一 GPU 提交，并不提供与线程数成比例的加速。该设备不支持 CUDA memory pools，兼容路径使用 `cudaMalloc`/`cudaFree`，失去异步池分配的复用优势；这些开销计入评估时间。尚未通过 profiler 区分密码运算、分配、传输和同步各自占比。百万 ID 场景的 44 位 rank 目前使用 FheUint128，进一步缩小类型可减少 radix 块，但尚未实现或测量该优化。

## 百万 gather 的容量检查

以同一 TFHE-rs 构建和默认参数实际加密、解密一条 FheUint128 记录，测得 64 个 radix 块 × 2049 个 u64，即 **1,049,088 字节/记录**。`N=1,000,000, C=500, L=2000, u=16` 的输入密文数据下界为 **1,049,088,000,000 字节（977.039 GiB）**；固定 32,000 条输出另需 **33,570,816,000 字节（31.265 GiB）**。

测量时主机可用 **242,869,129,216 字节（226.190 GiB）**，GPU 可用 **45,352,091,648 字节（42.237 GiB）**。当前 gather 接口全量持有独立输入密文，下界已超过主机容量，尚不含密钥、对象和工作区。容量检查状态为 **CAPACITY_BLOCKED**、退出码 77；**没有完整运行百万 gather，评估时间为 N/A**。8000 次相等比较、1600 万次 CMUX 是操作计数，不是实测时间。

复现：`bash experiment/tfhe_rs/check_gather_capacity.sh`。原始 [JSON](logs/tfhe_gather_1m_capacity.json) 与 [日志](logs/tfhe_gather_1m_capacity.log)。需要固定公开顺序的分批加载才能继续完整规模测试。

## TFHE 秘密簇选择与 gather

已实现 `src/he/tfhe_select.cpp` 的 `Evaluator::gather_clusters()`，由 `Server::gather_clusters()` 提供角色接口。处理顺序固定为 TFHE 质心 Top-u → TFHE 秘密 gather → TFHE 候选比较；新增的原生 C API 签名通过官方生成头文件的编译校验。加密簇 ID 只进入掩码和相等运算，全部簇都会扫描，输出固定为 `u*L` 条加密 `(distance,id)`。

| 本次单次 GPU 检查 | 结果 |
|---|---|
| 原有 Top-k 与新增 gather 组件 | 全部通过；27 次评估合计 **18764.173 ms**，墙钟 **20071.580 ms** |
| 紧凑记录：C=3、u=2、L=2 | gather **287.049 ms**；输出 4 条；质心 Top-u **309.602 ms**，候选 Top-k **326.722 ms** |
| 改变秘密选择、保持公开尺寸 | 输出随选择正确变化；完整逻辑地址序列相同，共 12 次记录选择 |
| gather 边界 | `u=1`、`u=C`、并列质心、簇首尾、32/128 位组合、无效/不存在的选择 ID、非法公开布局与输入不变均通过 |
| 真实 BFV 转换输出 → Top-u → gather → Top-k | 完整 23 位、FheUint128；C=3、u=2、L=2；输出 4 条距离及原始 ID 正确 |

转换接入样例先转换 11 个边界距离，取前三个作为质心、其后六个作为三个候选簇。新增路由链路耗时为：质心 Top-2 **604.158 ms** + gather **671.709 ms** + 候选 Top-3 **673.400 ms** = **1949.267 ms**。此时间排除 BFV 距离、转换、密钥生成、加解密以及原有 Top-3 of 11 回归调用。当前测试使用固定合成数据，不是聚类质量或完整 ANN 测试。

GPU 全目标及 CPU 逻辑构建通过。日志：[GPU 构建](logs/tfhe_gather_build.log)、[Top-k/gather 组件](logs/tfhe_gather_component.log)、[真实转换接入](logs/tfhe_gather_converted.log)、[CPU 逻辑](logs/tfhe_gather_cpu_checks.log)。每个配置只执行一轮；未重跑百万距离基准或 500 元素转换基准，也未运行百万条 gather。当前 gather 接口接收完整 TFHE 簇数组，大规模内存及公开分批加载尚需处理。

访问诊断记录实际 CMUX 调用位置的源/目标逻辑索引，按公开任务序号存放，比较不同秘密选择时的完整序列；不记录线程物理交错，也不构成底层 GPU/密码库的全面侧信道证明。接口和固定扫描说明见 [OBLIVIOUS_SELECT.md](docs/OBLIVIOUS_SELECT.md)。

## 代码整理后的回归结果

本次整理将实体移至 `src/he/entity/`；加解密、转换以及 `types.hpp`、`parameters.hpp` 移至 `src/utils/`。`he` 顶层只保留 `bfv_distance`、`tfhe_select`、`tfhe_topk`，内部网络和不透明 TFHE 类型放在 `he/detail/`。Client 只负责密钥和上下文，Server 复用评估对象；BFV 和 TFHE 加解密工具均验证了 Client 释放后仍可使用其共享密钥。

| 本次检查 | 结果及日志 |
|---|---|
| GPU 全目标及官方 C ABI 校验 | [构建通过](logs/refactor_build.log)，包含两个独立实验程序 |
| Rust 转换桥从新目录重新构建 | 通过；`third_party/build_bfv_tfhe_bridge.sh` 已同步路径 |
| 独立 CPU 构建与网络/参数检查 | [通过](logs/refactor_cpu_checks.log) |
| 1,000,000 × 128D uint8 BFV 距离 | [全部通过](logs/refactor_bfv_distance.log)；百万条评估 **2729.010 ms** |
| TFHE Top-k 的全部现有组件样例 | [通过](logs/refactor_tfhe_topk.log)；7 次评估合计 **13170.208 ms** |
| GPU 转换：11 个完整精度边界 | [11/11 通过](logs/refactor_conversion_full.log)；FheUint128；转换 **3639.822 ms**；转换后 Top-3 通过 |
| GPU 转换：11 个截断边界 | [11/11 通过](logs/refactor_conversion_drop7.log)；FheUint32；转换 **3103.513 ms**；转换后 Top-3 通过 |

每种回归配置只运行一次；本次没有重跑 500 个元素的转换性能基准或 100 选 10 性能基准，之前的测量保留在下文。调用方式见 [README](README.md#代码结构与职责)。

## 转换实现阶段的测量（重构前）

`test/` 保持用户指定的四个文件：`check.cpp`、`bfv_distance_test.cpp`、`bfc_tfhe_conversion.cpp`、`tfhe_topk.cpp`。共享计时在 `src/utils/check_support.hpp`，额外探索代码在 `experiment/`。

| 本轮检查 | 结果 |
|---|---|
| C++/CUDA 全部目标、官方 TFHE C ABI 编译校验、Rust 转换适配库 | 编译通过 |
| `check --cpu` | 通过；分组网络、100/10、500/16、并列、补齐、参数与位宽边界 |
| 新公式的 1,000,000 × 128D uint8 BFV 距离 | 全部精确，服务器评估 **2683.661 ms** |
| 单独 BFV 系数→LWE 提取 | 500/500 完整值正确，**322.098 ms** |
| CPU 分解 + GPU 整数修正的转换基线 | 500/500 截断分数与 ID 正确；转换后原生 GPU Top-3 通过；**325815.626 ms** |
| GPU 批量转换，完整 23 位边界测试 | 11/11 精确，转换后 Top-3 通过；**3589.881 ms** |
| GPU 批量转换，500 个距离、丢弃低 7 位 | **500/500 正确，输出 FheUint32，转换 91126.039 ms** |
| GPU 转换后原生 Top-3 of 11 | 距离、ID、并列规则正确，**1633.693 ms** |

CPU 对照与 GPU 主测试各运行一轮；迁移 GPU 前后各有一组 11 个完整精度边界检查。没有重复跑同一个大规模性能样例。该阶段 Top-k 算法未改动，使用实际转换输出检查互通；当时没有重跑 100/10 性能基准或全部独立 Top-k 样例。

## GPU 转换并行与时间

- 同一轮把 500 个 LWE 同时交给 TFHE-rs 的 CUDA key switching/PBS；24 轮 bit extraction 按依赖顺序执行。
- 后续原生整数模数修正、右移和附加 ID，通过 4 个 CPU 工作线程及各自 CUDA stream 在 GPU 上执行。
- 前面的 RNS/GMP 系数重构仍在 CPU。当前是 GPU 承担主要密码运算的转换实现，并非所有步骤都在 GPU。
- CPU/GPU 混合基线 **325.816 s** → GPU 批处理版 **91.126 s**，整体约 **3.58 倍**；其中密钥切换/分解由 **247.488 s** → **41.410 s**，整数修正及输出由 **77.913 s** → **49.317 s**。整体收益包含 GPU 批处理与末尾 4 stream 并发两项改动。
- GPU 500 测试一次性初始化共 **8591.139 ms**，其中转换评估材料生成 **7778.545 ms**（以原始日志为准）；输入加密 **24.298 ms**，客户端验证 **99.663 ms**，组件墙钟 **102129.732 ms**。
- 该转换样例的真实链路为：一批 4096 条 BFV 距离 **13.182 ms** + 选中 500 个转换 **91126.039 ms** + 其中 11 个的 Top-3 **1633.693 ms** = **92772.915 ms**。独立 LWE 提取诊断、初始化和客户端操作不计入这个和。它不是 500 选 10，也不是完整 ANN 延迟。

原始记录：[GPU 500](logs/bfv_tfhe_gpu_500.log)、[GPU 完整精度边界](logs/bfv_tfhe_gpu_smoke_full.log)、[CPU/GPU 基线](logs/bfv_tfhe_native_500.log)、[CPU 完整精度边界](logs/bfv_tfhe_native_smoke_full.log)、[汇总 JSON](logs/bfv_tfhe_conversion_summary.json)、[CPU 逻辑 CSV](logs/bfv_tfhe_gpu_cpu_checks.csv)。

## 距离公式和精度

正式实现使用 `Enc(||q||²) - 2Σ Enc(q[d])·x[d] + ||x||²`，数据库按多项式系数打包。客户端一次加密 128 个坐标和一个平方范数，服务器无需 relin 或旋转密钥。默认仍为 `N=4096, t=8380417`（23 位模数）。

百万条距离共 245 批，末批 576 条真实记录；全部记录、补齐、查询坐标及范数复用不变均通过。7 次回归覆盖单维、三维、最大距离与最大范数相消；还检查越界输入。百万条服务器评估 **2683.661 ms**，约 **372625 条/秒**；含回归全部 252 次 BFV 评估为 **2718.328 ms**，组件墙钟 **4025.566 ms**。见 [日志](logs/bfv_linear_coefficient_1m.log)。

`utils::BfvToTfheConverter::convert(..., id_bits, 0)` 保留完整距离；设为 7 时在密文上计算 `distance >> 7`，与 CPU 参考一致。后者对 500 个候选需 `16+9+1=26` 位 rank，使用 `FheUint32`。截断会增加并列，可能改变完整精度 Top-k；测试中 127/128/129 分别变为 0/1/1。完整 23 位路径目前验证 11 个边界，500 个主测试验证的是截断后的 16 位值。

转换密钥只在客户端使用两方秘密生成；Server 持有的 `BridgeEval` 没有秘密密钥字段。服务器不解密、没有回调客户端、没有从明文距离重新加密。自定义高精度 PBS 分解参数已通过上述功能测试，**尚未证明整条转换的组合失败概率**，属于实验实现。详细数学条件和参数见 [转换文档](docs/BFV_TO_TFHE.md)。

## 运行和报告

```bash
cd /home/student.unimelb.edu.au/xinyutu/PrivateImpl
bash third_party/build_bfv_tfhe_bridge.sh
./build/privateimpl/bfc_tfhe_conversion
# Optional full-precision boundary check:
./build/privateimpl/bfc_tfhe_conversion --smoke --full-precision
```

默认是 GPU 批处理。显式设置 `PRIVATEIMPL_CONVERSION_BACKEND=cpu` 可选 CPU 分解参考；该模式的末尾整数运算同样支持 4 个 GPU 工作线程，历史基线的末尾提交当时为串行。

总入口 `./build/privateimpl/check` 依次执行组件，并把结果写入 `logs/evaluation_times.csv`；`--report PATH` 可修改输出。CSV 分别记录距离、转换、源距离和转换后 Top-3 时间；gather 和桶内最小值调用计入组件总评估时间并单独打印阶段耗时。返回 0=请求的组件全通过，1=失败，2=有不可用组件。组件通过不代表完整 ANN 已完成：输出净化及 SIFT/DEEP1M 数据集集成/召回率实验仍缺失，因此完整 ANN 时间为 N/A。连续到桶优胜者 Top-k 的实验时间单独保存在 `selection_pipeline_once.csv`。

本轮验证使用 NVIDIA L40S-48Q，memory pools supported=no；已有 CUDA 分配兼容补丁继续生效。转换适配库安装在 `third_party/tfhe-rs-install/lib/libbfv_tfhe_bridge.so`，构建复用本地 TFHE-rs rlib 和依赖，无须下载新库。

## 历史测量与诊断

旧 CRT 距离的百万条时间为 **2666.354 ms**，不能与新系数打包混写为同一实现。旧 CRT 槽提取在 4096/23-bit 下失败；已定位为明文 mask 乘法耗尽噪声余量，新系数打包直接提取避开该操作。旧日志和 `experiment/bfv_conversion/diagnose.cpp` 保留作原因复现；正式转换测试已接通原生 TFHE-rs。

以下 OpenFHE 与 Top-k 测量属于此前各轮版本；其中的 OpenFHE 互通限制仍适用于该独立实验，本次 BFV 转换接口不接受 OpenFHE 输入。

## 独立 OpenFHE 原生接口实验

OpenFHE `v1.6.1`（commit `b59e1f291965f14fbee5172cd83a20f5b856204f`）已安装到 `third_party/openfhe-install/`，上游源码未修改。实验源码位于 `experiment/openfhe/`，不使用项目实体封装，`test/` 仍为四个文件。

- **原始转换示例**：`scheme-switching` 退出码 0，整体 21.37 秒；包含多个演示和密钥生成，部分参数为 TOY。这不是百万条转换时间，也不代表 TFHE-rs 格式互通。
- **原生 CKKS 明密文乘法**：CPU、8 个 OpenMP 线程，1,000,000 条 × 128 维 uint8，共 128,000,000 个乘积；全部逐项验证通过，查询未被修改。
- **乘法总时间 1.785455 秒**；明文编码 8.135106 秒；两者合计 **9.920561 秒**。含客户端解密验证的测试墙钟时间为 60.405517 秒。
- CKKS 参数为 `HEStd_128_classic`、深度 1、环维度 16384、8192 槽位、scale 40 bit、first modulus 60 bit；最大绝对误差 `6.34917014395e-05`，所有乘积舍入后等于完整整数结果。
- 65 条向量的先行检查覆盖最大乘积、零值及末批补齐。初始 50-bit scale 不满足该输入范围，调整为上述参数后通过。

完整数据与复现命令见 [OpenFHE 实验说明](experiment/openfhe/README.md)，原始结果见 [CSV](logs/openfhe_plain_multiply_1m_128_u8.csv)。这些时间不包含 CKKS→FHEW 转换，也没有执行平方 L2 或 Top-k。

## CKKS → OpenFHE FHEW → TFHE-rs，500 个质心距离

新增独立实验 `experiment/openfhe/run_tfhe_interop.py`，`test/` 仍保留四个原入口。明文计算 500 × 128 维 uint8 平方 L2，完整距离加密为 CKKS，再调用原生方案转换，随后尝试导入 TFHE-rs 1.5.4。

- **底层 LWE 导入通过**：两组各 500 个密文全部保持与 OpenFHE 相同的解密结果。仅做公开的系数提升和布局适配，没有解密后重新加密。
- **GPU 整数接入未完成**：原生 conformance 检查各 0/500 符合当前 `FheUint128` 输入要求；n=1305 与目标 2048 不同，单 LWE 大整数与 64 个 radix 块不同，密钥及 padding/carry 编码也未转换。GPU `min/max` 未执行。
- **完整 23 位距离精度失败**：沿用示例 q=2^25、中间模数 27 位，指定 p=2^23，本次 65/500 完全正确；转换耗时 174.000 ms，不能作为有效完整链路性能。
- **模意义对照通过**：相同 CKKS 输入，使用示例 p=q/(2*beta)=2^17，500/500 正确等于原距离 mod p；转换耗时 169.170 ms，但取模不保持原距离大小关系。
- q=2^31 的原生参数生成调用实际报错 `logQ > 29 is not supported`。这里只报告本次参数路径限制，没有断言所有自定义方案都不可行。
- 脚本退出码 **2（BLOCKED）**；完整链路总评估时间保持 **N/A**。两组实验墙钟约 951.665 ms，排除编译，包含一次公共初始化和客户端校验；随机加密使失败计数和计时随运行变化。

复现：`python3 experiment/openfhe/run_tfhe_interop.py`。详细参数、计时定义和桥接障碍见 [实验说明](experiment/openfhe/README.md)，结果见 [汇总 JSON](logs/openfhe_tfhe_interop_summary.json) 和 [TFHE-rs 实测日志](logs/tfhe_rs_openfhe_import_500.log)。

## OpenFHE CPU max / 100 选 10

新增独立实验 `experiment/openfhe/ckks_topk_cpu.cpp`，使用 OpenFHE 原生 `EvalMaxSchemeSwitching`、`EvalMinSchemeSwitching` 和 `EvalCompareSchemeSwitching`。原生库有 min/max/argmin/argmax 混合接口，没有找到现成 Top-k；本实验用 10 轮原生 argmin 和加密排除掩码组合出 Top-k，中间不解密。

- 输入：100 个 128 维 uint8 质心的明文平方距离，缩放 `1/65536` 后加密为 CKKS；无整数截断，补齐为 128 槽。
- 参数：CKKS `HEStd_128_classic`、N=65536、深度 23、scale 50、first modulus 60、4 个大数位；FHEW `STD128`、n=1305。CPU 8 个 OpenMP 线程。
- **argmax 位置和 Top-10 全部位置按顺序验证通过**，输入未修改，补齐项未入选。返回位置 `[0, 28, 49, 9, 48, 36, 26, 85, 32, 61]`。
- 独立 max/argmax 调用 **69.715937 秒**；Top-k 的 10 次 argmin 共 691.538304 秒，9 次加密掩码刷新/更新共 370.008102 秒，**Top-k 总评估时间 1061.546406 秒，约 17 分 42 秒**。
- 上下文及密钥生成 5.839835 秒；整个实验墙钟 1137.991270 秒，包含独立 max、初始化、加密与验证。进程退出码 0，峰值 RSS 约 11.63 GiB。
- **数值限制**：返回 CKKS 距离的最大绝对误差为 9386.357146；位置排序通过不表示完整距离数值精确。独立 max 的正确性检查针对 argmax 位置。本次不是任意 23 位输入的精确比较保证，也没有覆盖并列情况。
- 这是一次 CKKS↔FHEW 混合接口的逐轮 argmin 基线，不是原生纯 TFHE Top-k，也不代表最快实现。该实验没有改变此前 TFHE-rs 桥接未完成的状态。

完整参数、命令和计时边界见 [实验说明](experiment/openfhe/README.md)，数据见 [CSV](logs/openfhe_topk_cpu_100_10.csv)、[原始日志](logs/openfhe_topk_cpu_100_10.log) 和 [汇总 JSON](logs/openfhe_topk_cpu_100_10_summary.json)。

## TFHE-rs GPU：单次 100 选 10

2026-10-12，按用户要求只执行一次 Top-k，无预热或重复测试。独立目标 `tfhe_topk_once` 复用现有选择网络及 TFHE-rs 原生 GPU `min/max`，不初始化 BFV；`test/` 仍为四个文件。

- TFHE-rs 1.5.4、NVIDIA L40S-48Q；100 个 128 维 uint8 向量的完整平方距离，数据与此前 OpenFHE CPU 实验一致。
- 沿用 `FheUint128` 距离/编号编码，补齐至 128 个候选，1,483 个比较节点、2,966 次原生 `min/max` 调用。
- **服务器 Top-k 评估时间 332.928619 秒，约 5 分 33 秒**；调用前后同步 GPU，排除密钥生成、加解密和明文距离生成。
- 密钥生成 0.658488 秒，GPU 密钥加载 0.092502 秒，输入加密 0.727734 秒；整个进程 334.77 秒。
- **10/10 个距离和编号精确校验通过**，退出码 0。输入直接加密为 TFHE-rs；未执行 CKKS→TFHE-rs 转换，桥接状态仍未完成。

详见 [单次实验报告](experiment/tfhe_rs/README.md)、[CSV](logs/tfhe_rs_topk_100_10_once.csv) 和 [日志](logs/tfhe_rs_topk_100_10_once.log)。

## TFHE-rs GPU：迁移 CPU 分组裁剪与比较复用

2026-10-12，已将 `PrivateANN/single-server/single-server-tfhe-topk-radix` 的公开网络调度移植到 `src/he/detail/topk_schedule.hpp`，接入现有 `he::topk`。双输出比较节点使用一次原生 `gt` 加两次原生 `if_then_else`，单输出节点只算原生 `min`；服务器不解密，也不需要第二个服务器。

- 同样的 100/10 数据、L40S-48Q 和 `FheUint128` 编码，节点由 1,483 减少到 729（633 双输出、96 单输出）。
- **仅复测一次：Top-k 服务器评估 109.780647 秒**，相对原始 332.928619 秒约快 **3.03 倍**；整个进程 111.60 秒。
- **10/10 个距离和编号精确校验通过**，退出码 0。C API ABI 编译校验、CPU 网络/并列/边界/补齐/层内独立性检查通过。未重跑完整加密回归套件。
- CPU 同层并行与 lazy LUT 优化未迁移，位宽未调整；输入仍直接加密为 TFHE-rs，转换桥接仍未完成。

实测结果及剩余优化点见 [报告](experiment/tfhe_rs/README.md)、[CSV](logs/tfhe_rs_topk_100_10_grouped_once.csv) 和 [日志](logs/tfhe_rs_topk_100_10_grouped_once.log)。

## TFHE-rs GPU：FheUint32 与同层 4 线程并行

2026-10-12，100/10 输入改用无截断的 23 位距离 + 7 位 ID + 1 位补齐标记，直接加密为 `FheUint32`。更大公开位宽自动保留宽整数，同批编码不一致或字段越界会报错。

- 同层节点通过 4 个 CPU 工作线程和各自 CUDA stream 提交到同一 L40S，层间同步；密文比较和选择仍由 TFHE-rs GPU 执行。线程复用整轮 Top-k 的 stream。
- **单次 100/10 服务器评估 41.865028 秒**，上一版 109.780647 秒，约快 **2.62 倍**；相对最初 332.928619 秒约快 **7.95 倍**。这是位宽、并行和 stream 复用的组合效果，未分别做性能采样。
- **10/10 个距离和编号精确正确**；实际工作线程 4 个，最多 4 个并发算子调用。该计数不代表已测得 GPU kernel 的重叠程度。
- 密钥生成 0.656249 秒、GPU 密钥加载 0.103126 秒、加密 0.183004 秒，均单独计时；整个进程 43.15 秒，退出码 0。
- CPU 编码及网络检查、C API ABI 编译检查通过；`tfhe_topk` 组件的一轮 7 个小样例全部通过，覆盖紧凑编码 bit-31 补齐、并列、宽整数边界、非法 k、混合编码拒绝和输入不变性。本轮未重跑 BFV 百万条距离或转换实验。

详见 [报告](experiment/tfhe_rs/README.md)、[CSV](logs/tfhe_rs_topk_100_10_u32_parallel_once.csv)、[GPU 日志](logs/tfhe_rs_topk_100_10_u32_parallel_once.log) 和 [组件检查](logs/tfhe_u32_parallel_component_check.log)。

## TFHE-rs GPU：128 位并行对照

按用户要求评估保留 128 位编码的收益。只增加实验入口的 `OUTPUT_CSV 128` 选项，选择原始 `(distance << 32) | ID`、`2^64` 补齐和 4 个工作线程。

- 单次 100/10、相同 L40S-48Q、729 个节点和输入：**服务器评估 95.150926 秒**，原 128 位串行版本 109.780647 秒。
- 耗时下降 **13.33%**，约 **1.15 倍**；收益包含并行调度和 CUDA stream 复用。此前 32 位并行版本为 41.865028 秒。
- **10/10 个距离和编号精确正确**，4 个实际工作线程，最多 4 个并发算子调用；整个进程 96.94 秒，退出码 0。
- 本次仅运行一次 128 位并行基准，没有重复运行另外两种配置。

数据见 [报告](experiment/tfhe_rs/README.md)、[CSV](logs/tfhe_rs_topk_100_10_u128_parallel_once.csv) 和 [日志](logs/tfhe_rs_topk_100_10_u128_parallel_once.log)。
