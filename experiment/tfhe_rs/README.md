# TFHE-rs GPU 实验

当前表示为独立距离/ID 密文，只比较距离，并列可任选；最新组件与容量测量见 [TEST_STATUS](../../TEST_STATUS.md)。连续链路默认输出 `logs/selection_pipeline_split_once.csv`，历史 packed rank CSV 保留。

## 连续显存与 CPU 调度线程实验

```bash
cmake --build build/privateimpl --target bucket_max_once -j 4
PRIVATEIMPL_TFHE_ARENA_MB=8192 ./build/privateimpl/bucket_max_once 2 300 4 logs/my_max_2x300.csv
PRIVATEIMPL_TFHE_ARENA_MB=8192 ./build/privateimpl/bucket_max_once 10 300 4 logs/my_max_10x300.csv
```

参数依次为桶数、每桶容量、CPU 调度线程数、报告路径。省略 arena 环境变量使用原始分配路径；将第三个参数改为 1 或 10 可检查主机提交并发。所有记录有效，距离和 ID 分别为 FheUint32，每个节点只比较距离，并根据密文条件选择成对字段。这里测最大值，不能直接当成已经重测的大规模最小值流水线。

单次结果：两个 300 条桶在原始分配/4 线程下为 39.787 s，预分配/4 线程为 22.725 s；预分配/1 线程为 51.236 s。十桶预分配/4 线程为 108.348 s，10 线程为 127.847 s，全部结果正确。默认线程数仍为 4。arena 同步回收还保留全设备等待，尚未实现整层批量 PBS；详细归属及限制见 [TEST_STATUS](../../TEST_STATUS.md)。

可选 CUPTI 探针依赖 NVIDIA `nvidia-cuda-cupti==13.0.85` wheel 的 `nvidia/cu13` 目录，解压位置为 `third_party/cupti-13.0/`；CMake 仅在该目录存在 `lib/libcupti.so.13` 时构建 `cupti_overlap_probe`。它不属于正常比较/选择的运行依赖。当前 vGPU 上取得的 kernel 时间戳全为零，因此此探针不能提供内核重叠率；大体积原始跟踪不随 Git 仓库分发。

## 小型桶并行诊断

```bash
cmake -S . -B build/privateimpl
cmake --build build/privateimpl --target bucket_parallel_once -j 4
./build/privateimpl/bucket_parallel_once
```

每种配置只执行一次，使用现有 `bucket_minima()`：单桶 300 取 1（299 个节点，4 线程），100 桶各 4 取 1（合计 300 个节点，分别用 4 / 16 线程）。距离和 ID 各 FheUint32，每项使用新加密输入，客户端验证所有桶的最小值及关联 ID。

L40S-48Q 本次分别为 **20.080 s / 20.642 s / 21.818 s**，全部通过。16 线程的峰值在途调用数确实为 16，但本轮未提高吞吐；单张 GPU 的有限资源不能按独立桶数等比例扩展。生产默认仍为 4。该诊断既不是重复采样统计，也没有重新运行百万/30k 基准。结果见 [CSV](../../logs/bucket_parallel_once.csv) 与 [报告](../../TEST_STATUS.md)。

只测两个桶各 300 取 1，可运行：

```bash
./build/privateimpl/bucket_parallel_once --two-buckets
```

此模式只执行一个配置：两桶同层合并、共享 4 个调度线程，598 个节点、9 层。单次 **38.196 秒**，两个桶的距离和 ID 均验证通过；约为此前单桶 20.080 秒的 **1.90 倍**，当前实现没有表现出明显的桶级吞吐提升。初始化和输入加密排除在评估计时外。默认输出 [bucket_two_300_once.csv](../../logs/bucket_two_300_once.csv)，也可以在开关之后给出自定义 CSV 路径。

### 定位 CUDA 调用开销

```bash
PRIVATEIMPL_PROFILE_BUCKETS=1 nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o logs/bucket_two_300_profile \
  ./build/privateimpl/bucket_parallel_once --two-buckets logs/bucket_two_300_profile.csv
```

该开关只标记服务器归约的采集边界。跟踪有额外开销，CSV 单独保存，不能替代普通运行的性能数据。一次跟踪在 598 个 minimum 节点中记录到 115,420 次 `cudaMalloc`、115,416 次 `cudaFree`、1,200 次 stream 创建，提供了工作区复用的检查方向。

当前 Nsight 与 driver 版本不兼容，采集结果缺少 GPU kernel activity，**无法计算内核重叠率**。`workers_used` 和 `peak_inflight` 都是主机侧诊断，不能称为 GPU 并行证明。详见 [摘要及限制](../../logs/bucket_two_300_profile_summary.json) 与 [测试报告](../../TEST_STATUS.md)。

### 核对 SM 并发能力与 TFHE 启动参数

```bash
cmake -S . -B build/privateimpl
cmake --build build/privateimpl --target bucket_parallel_once cuda_launch_probe cuda_overlap_probe -j 4
./build/privateimpl/cuda_overlap_probe
LD_PRELOAD="$PWD/build/privateimpl/libcuda_launch_probe.so" \
  PRIVATEIMPL_PROFILE_BUCKETS=1 \
  PRIVATEIMPL_LAUNCH_REPORT=logs/tfhe_launch_shapes.csv \
  ./build/privateimpl/bucket_parallel_once --launch-check
```

`cuda_overlap_probe.cu` 在两个非默认 stream 各启动一个 8-block 小内核，记录设备全局时间戳和 SM 编号。L40S-48Q 本次普通/cooperative 内核均在合计 16 个不同 SM 上观察到约 4.995 ms 的跨 grid block 时间重叠，每个 grid 跨度约 5 ms。全部内存在提交前分配；这是硬件能力检查，不能证明 TFHE 内核也同样重叠。

`cuda_launch_probe.cpp` 是只记录公开 grid/block/共享内存/stream 参数的 LD_PRELOAD 探针，原样转发 CUDA launch，不读取密文、不改变分配、不额外同步 GPU。`--launch-check` 只执行两个 4 条桶的 6 个 minimum 节点，保留完整 23 位距离和 20 位 ID，两个结果均验证通过。本次归约捕获 **3,252 次内核启动**，平均 **542 次/节点**；普通 grid 最多 1,856 blocks，cooperative grid 为 2–64 blocks。未观察到空 stream，多个形状各使用 4 个 stream。

启动参数不等于实际 SM 占用率；本探针没有 GPU 执行时间线，无法计算真实 TFHE 内核的重叠。拦截运行的 441.568 ms 含采集开销，不作为性能基线。生产代码仍按节点调用高层 TFHE 运算，尚未实现同层批量算子或工作区复用。结果：[摘要](../../logs/tfhe_launch_diagnosis.json)、[硬件时间戳](../../logs/cuda_overlap_probe.csv)、[launch 参数](../../logs/tfhe_launch_shapes.csv)、[正确性及计时](../../logs/bucket_launch_check.csv)。

## 单次大规模组件：500 选 15，30,000 条分 100 桶，最终 Top-10

```bash
cmake -S . -B build/privateimpl
cmake --build build/privateimpl --target cluster_bucket_once -j 4
./build/privateimpl/cluster_bucket_once
```

`cluster_bucket_once.cpp` 实际计算 1M 条 128 维 uint8 记录的 BFV 距离，另计算并转换 500 个质心，然后调用现有质心 Top-15。候选部分复用 `random_buckets`、`bucket_minima` 和 `topk`：30,000 条、100 桶各 300 条、100 个优胜者选最终 10 条。距离和 ID 各为 FheUint32，距离保留完整 23 位，ID 从客户端输入起就是 TFHE 密文。只执行一次，长阶段每 30 秒报告公开的耗时进度。

**这是组件测量，不是完整秘密查询。** 原有连续流水线先将所有距离转为 TFHE，然后调用 TFHE gather。新的“只转换质心，选簇后转换候选”顺序还缺少由 TFHE 选择结果控制的 BFV oblivious gather。本实验预先按公开合成数据的明文参考准备 15 个簇的 TFHE 候选输入；没有解密选择结果用于服务器路由，也没有执行候选 BFV→TFHE 转换。CSV 的 `full_pipeline_executed=0`、`candidate_conversion_executed=0` 明确记录该边界。组件时间之和不能当作完整查询延迟。

数据为 500 个公开合成代表向量及每簇 2,000 条附近记录，不包含聚类算法。最终校验全部 1M BFV 距离、500 个转换结果、质心 Top-15、100 个桶最小值及最终 Top-10；桶内并列允许任选，但距离与原始 ID 必须匹配。报告中的 recall 只度量该合成 30,000 候选集合的分桶损失，并在 Top-10 截止距离处允许并列。

默认报告为 `logs/cluster_bucket_once.csv`，可用唯一位置参数更改路径。输入生成、客户端加密、密钥初始化及解密验证均排除在服务器组件计时外；BFV 列准备另列，原生 GPU 调用的编码、数据传输及同步包含在对应操作时间内。

2026-10-13 单次 L40S-48Q 实测通过：百万 BFV 距离 **3.006 秒**，500 个质心转换 **74.494 秒**，500 选 15 **326.342 秒**，100 桶归约合计 **2112.137 秒**，最终 100 选 10 **65.758 秒**。已测服务器组件合计 **2581.752 秒（43.03 分钟）**；不含缺失的 BFV gather 和候选转换，不能作为完整查询时间。桶内归约及最终 Top-k 均为 4 个工作线程，峰值在途调用数 4。全部检查通过，合成候选集的 distance recall@10 为 **0.9**。详细数据见 [测试报告](../../TEST_STATUS.md) 和 [CSV](../../logs/cluster_bucket_once.csv)。

## 单次连续链路：BFV 距离 → 桶优胜者 Top-k

`selection_pipeline_once.cpp` 复用现有计算接口，连续执行 BFV 距离、真实 BFV→TFHE 转换、质心 Top-u、TFHE gather、公开随机位置分桶、并行桶内最小值、桶优胜者 Top-k。服务器调用之间不解密，结束后统一验证。

```bash
cmake --build build/privateimpl --target selection_pipeline_once -j 8
./build/privateimpl/selection_pipeline_once
# Optional report path:
./build/privateimpl/selection_pipeline_once logs/my_pipeline.csv
```

每次命令运行一轮，无预热或重复统计；后面的命令只是自定义报告路径的替代写法。100 条 128D uint8 记录、10 个合成簇各 10 条、u=4、B=16、k=10，完整 23 位距离及 20 位 ID，当前分别使用 FheUint32。质心取簇内量化均值，110 个距离全部由 BFV 计算并转换。

下面是字段分离前的 packed FheUint128 历史实测，当前命令不会复现旧布局：服务器 **54.533 秒**，含初始化/客户端验证的实验墙钟 **63.522 秒**；桶内归约 **2.760 秒**，桶优胜者 Top-k **8.296 秒**。两者均记录 4 个工作线程、4 个同时在途的算子调用。结果与明文分桶参考完全一致，对全量精确 Top-10 的单种子 recall 为 0.8。详细参数与全部阶段见 [TEST_STATUS](../../TEST_STATUS.md)，[CSV](../../logs/selection_pipeline_once.csv) 与 [日志](../../logs/selection_pipeline_once.log)。

范围截至加密 Top-k，尚未实现输出净化、ID-only 响应和网络传输。公开随机分桶在 CPU 重排密文句柄，GPU 执行距离/比较/选择；转换含 CPU 提取步骤。此次是小规模合成数据的连续测试，百万规模容量问题仍见 [oblivious select](../../docs/OBLIVIOUS_SELECT.md)。

## 独立单次 100 选 10

`topk_once.cpp` 使用现有 `he::topk` 的选择网络和 TFHE-rs GPU 原生接口，只调用一次 Top-k，无预热、无重复轮次，也不初始化 BFV。

- 100 个 128 维 uint8 向量的平方 L2 距离，固定种子 `20261012`，与 OpenFHE CPU 实验的数据相同。
- 距离为完整整数，最多需要 23 位；当前用独立的 `FheUint32` 保存距离和编号，距离哨兵为 `2^23`，无截断。输入直接加密为 TFHE-rs 密文。
- 当前网络从单服务器 CPU 原型移植：按 `next_power_of_two(k)` 分组排序，逐轮裁剪合并。100/10 补齐至 112 个候选，25 层；633 个双输出节点各调用一次距离 `gt` 和四次字段 `if_then_else`，96 个单输出节点各调用一次距离 `gt` 和两次字段选择。默认 4 个 CPU 工作线程并行提交同层 GPU 运算，层之间同步。
- `topk_evaluation_ms` 在调用前后同步 GPU，包含补齐、主机调度、数据传输和全部 GPU 评估；不包含密钥生成、GPU 密钥初始化、客户端加解密和明文距离生成。
- 只在评估结束后解密，检查 10 个距离顺序正确、编号对应原始记录，允许任选并列记录。
- 本实验没有执行 CKKS→TFHE-rs 转换；此前转换接口尚未接通。

在项目根目录构建和运行（使用 CMake 3.30.4 或更高版本）：

```bash
cmake -S . -B build/privateimpl
cmake --build build/privateimpl --target tfhe_topk_once -j 4
CUDA_VISIBLE_DEVICES=0 ./build/privateimpl/tfhe_topk_once
```

当前版本默认 CSV 为 `logs/tfhe_rs_topk_100_10_split_once.csv`。距离和 ID 分别使用 FheUint32，只有距离参与比较；并列允许任选。旧的 `32|128` 布局参数已移除，下面各节保留修改前的历史数据。

## 2026-10-12 原始基线

旧实现补齐至 128 个候选，28 层、1,483 个比较节点，共 2,966 次原生 `min/max` 调用；每次独立计算 min 和 max，没有复用比较结果。下面的数据对应修改前的实现。

CSV：[基线单次计时](../../logs/tfhe_rs_topk_100_10_once.csv)。输出：[原始日志](../../logs/tfhe_rs_topk_100_10_once.log)。

TFHE-rs 1.5.4，NVIDIA L40S-48Q，单次运行退出码 0。

| 阶段 | 耗时 |
| --- | ---: |
| 客户端密钥生成 | 0.658488 秒 |
| GPU 评估密钥加载 | 0.092502 秒 |
| 100 个输入加密 | 0.727734 秒 |
| **Top-k 服务器总评估** | **332.928619 秒（约 5 分 33 秒）** |
| 结果解密和校验 | 0.004033 秒 |
| 整个进程墙钟 | 334.77 秒 |

10/10 个距离及编号完全正确，顺序为 `[0, 28, 49, 9, 48, 36, 26, 85, 32, 61]`。

此前 OpenFHE CPU 8 线程的同数据 Top-k 为 1061.546406 秒，本次耗时约为其 31.4%。两次实验的硬件、算法和密文表示不同：OpenFHE 使用 CKKS↔FHEW 混合接口逐轮 argmin，本次使用直接加密的整数和 GPU 选择网络。因此该比例只比较这两次具体实现，不是库本身的普遍加速比。

## 2026-10-12 迁移单服务器 CPU 的网络和比较节点

参考文件：`PrivateANN/single-server/single-server-tfhe-topk-radix/src/network.rs` 和 `he_ops.rs`，commit `79389cd5f349bc67a9c4e69945927c00a27f4fc1`。这些优化只使用公开网络结构和服务器评估密钥，不依赖双服务器协议。

迁移内容：

1. 按 `next_power_of_two(k)` 分组排序，然后逐轮裁剪合并，保留每组可能入选的较小部分。
2. 只需要最小值的节点只执行原生 `min`。
3. 双输出节点计算一次 `swap = gt(a,b)`，复用它计算 `if_then_else(swap,b,a)` 和 `if_then_else(swap,a,b)`。条件保持加密。GPU 比较和条件选择均使用 [TFHE-rs 原生接口](https://docs.zama.org/tfhe-rs/hardware-acceleration/run-on-gpu/gpu-operations)。

| 项目 | 原基线 | 迁移后 |
| --- | ---: | ---: |
| 输入补齐数 | 128 | 112 |
| 网络层数 | 28 | 25 |
| 比较节点数 | 1,483 | 729（633 双输出 + 96 仅最小值） |
| 节点内部排序比较次数 | 2,966 | 729 |
| **服务器 Top-k 时间** | **332.928619 秒** | **109.780647 秒** |
| 整个进程墙钟 | 334.77 秒 | 111.60 秒 |
| 距离和编号精确校验 | 10/10 | 10/10 |

同一 L40S-48Q、同样的固定数据、相同 `FheUint128` 编码和默认加密参数。迁移后仅运行一次加密 Top-k，无预热和重复采样；两次实测的耗时比为 **3.03**，下降约 **67.0%**，不提供重复采样意义上的统计保证。迁移后密钥生成 0.656476 秒、GPU 密钥加载 0.102783 秒、输入加密 0.732452 秒、结果解密及校验 0.005294 秒；这些均不计入服务器评估时间。

验证：构建时校验 C API 函数签名；`check --cpu --report logs/tfhe_grouped_cpu_checks.csv` 通过网络排序、并列、补齐、32 位边界及同层无冲突检查。加密实测退出码 0，10 个距离和编号均与独立明文排序一致。本轮没有再次执行整个加密回归套件。

这一阶段保留 128 位通用编码，未迁移同层线程并行和自定义 lazy LUT/noise 优化。下一节记录随后完成的 32 位编码与同层并行。

优化结果：[CSV](../../logs/tfhe_rs_topk_100_10_grouped_once.csv)、[日志](../../logs/tfhe_rs_topk_100_10_grouped_once.log)、[进程耗时](../../logs/tfhe_rs_topk_100_10_grouped_once.time)。

## 2026-10-12 FheUint32 与同层并行

当时的实现按公开的 `score_bits` 和 `id_bits` 选择密文类型：总位宽加补齐位不超过 32 时使用原生 `FheUint32`，否则使用 `FheUint128`。每个字段最多 32 位；超界输入和同批不同编码会报错。100/10 使用 `rank = (distance << 7) | ID`，补齐值为 `1 << 30`。500 个候选若声明 23 位距离和 9 位 ID，带补齐位需 33 位，会保留宽整数。

服务器默认启用 4 个 OpenMP 工作线程。每个线程为整轮 Top-k 设置一次 CUDA server key，原生 C API 克隆 key 时创建该线程使用的独立 CUDA stream，底层评估密钥共享。每层各节点读写互不重叠的位置；层结束后统一等待，工作线程异常在完成同步后回传。构造 `tfhe_rs::Evaluator(key, 1)` 可选择串行，默认参数为 4。

### CPU 线程与 GPU 执行的关系

这是 **TFHE-rs GPU 后端**：CPU 线程负责调度和调用 C API，`gt`、`min`、`if_then_else` 通过 CUDA server key 在 L40S 上计算。4 个 CPU 工作线程对应 4 个独立的顶层 CUDA 任务队列，每个 GPU 算子内部使用大量 CUDA 线程。TFHE-rs 官方说明了这种[通过 CPU 线程和 CUDA stream 并发调度 GPU 运算的方式](https://docs.zama.org/tfhe-rs/hardware-acceleration/run-on-gpu/multi-gpu)。

日志中的 `peak_inflight_gates=4` 统计尚未返回的并发算子调用，不是 GPU kernel 重叠程度或 GPU 利用率；本次没有使用 profiler 测量 kernel 重叠，也没有单独测定并行带来的加速比例。

### 单次结果

同一 L40S-48Q、相同 100 个距离和默认加密参数，当前版本只运行一次 100 选 10 性能测试，无预热和重复采样。

| 版本 | 密文位宽 | 同层工作线程 | Top-k 服务器评估 |
| --- | ---: | ---: | ---: |
| 原始 bitonic 基线 | 128 | 1 | 332.928619 秒 |
| 分组裁剪、复用比较 | 128 | 1 | 109.780647 秒 |
| **紧凑编码与同层并行** | **32** | **4** | **41.865028 秒** |

本次相对上一版约快 **2.62 倍**，相对原始基线约快 **7.95 倍**。本次同时修改位宽、线程调度和 CUDA stream 复用，因此这些比例是组合效果，不能分别归因于某一项。

- **10/10 个距离和编号精确校验通过**，顺序与此前两轮相同；退出码 0。
- 实际工作线程 4 个，最多 4 个并发算子调用；网络仍为 25 层、729 个节点。
- 客户端密钥生成 0.656249 秒、GPU 密钥加载 0.103126 秒、100 个输入加密 0.183004 秒、解密及校验 0.002433 秒；整个进程 43.15 秒。
- 服务器时间包含线程调度、线程 CUDA stream 初始化、数据传输、原生 GPU 运算和同步，排除上述客户端阶段及公共 GPU 密钥加载。
- 必要的正确性检查另运行了一次 `tfhe_topk` 组件：7 个不同小样例均通过，覆盖并列、输入不变、bit-31 补齐、宽整数兼容、非法 k 和混合编码拒绝；不是重复采样 100/10。CPU 编码和网络检查、生成头文件 ABI 编译检查也通过。
- 仍直接加密为 TFHE-rs，未执行 CKKS→TFHE-rs 转换；CPU 自定义 lazy LUT 优化未迁移。

结果：[CSV](../../logs/tfhe_rs_topk_100_10_u32_parallel_once.csv)、[GPU 日志](../../logs/tfhe_rs_topk_100_10_u32_parallel_once.log)、[进程耗时](../../logs/tfhe_rs_topk_100_10_u32_parallel_once.time)、[组件检查](../../logs/tfhe_u32_parallel_component_check.log)。

## 保留 FheUint128，仅加入同层并行与 stream 复用

以下为历史实验，旧位宽参数当前已移除。当时针对“不改成 32 位还能提升多少”，实验入口增加可选位宽参数。选择 128 会使用原始 `rank = (distance << 32) | ID` 和 `2^64` 补齐，输入数据、729 个节点和加密参数与 128 位串行基线一致。

```bash
# Historical command for the old binary; unsupported by the current split layout:
# CUDA_VISIBLE_DEVICES=0 ./build/privateimpl/tfhe_topk_once logs/tfhe_rs_topk_100_10_u128_parallel_once.csv 128
```

单次运行，无预热和重复采样，L40S-48Q：

| 配置 | Top-k 服务器评估 |
| --- | ---: |
| 128 位，原串行版本 | 109.780647 秒 |
| **128 位，4 个工作线程与 stream 复用** | **95.150926 秒** |
| 32 位，4 个工作线程与 stream 复用 | 41.865028 秒 |

保留 128 位时，本次相对串行基线节省 **14.629721 秒（13.33%）**，耗时比约 **1.15 倍**。这是同层并行和 stream 复用的组合收益；这组单次数据中，32 位编码带来的差异更大。128 位并行版本约为 32 位并行版本耗时的 2.27 倍。

- **10/10 个距离和编号精确校验通过**；实际工作线程 4 个，最多 4 个并发算子调用，退出码 0。
- 客户端密钥生成 0.653434 秒、GPU 密钥加载 0.093228 秒、输入加密 0.726794 秒、解密及验证 0.003522 秒，均排除在服务器计时外；整个进程 96.94 秒。
- 此次只执行 128 位并行基准一次，没有重跑原串行版本或 32 位版本。因此结果是单次观测，不是重复采样后的统计结论。

结果：[CSV](../../logs/tfhe_rs_topk_100_10_u128_parallel_once.csv)、[GPU 日志](../../logs/tfhe_rs_topk_100_10_u128_parallel_once.log)、[进程耗时](../../logs/tfhe_rs_topk_100_10_u128_parallel_once.time)。
