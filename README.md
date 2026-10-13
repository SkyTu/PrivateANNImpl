# PrivateImpl 当前实现

任务定义见 [prompt.md](prompt.md)，实际测试状态见 [TEST_STATUS.md](TEST_STATUS.md)。

## 从仓库恢复依赖

仓库包含项目代码、测试、实验报告、固定依赖版本及本地补丁；下载的第三方源码、工具链、已安装库、构建目录和大型 profiler 文件不入库。

```bash
git clone https://github.com/SkyTu/PrivateANNImpl.git
cd PrivateANNImpl
bash third_party/install.sh
```

此命令检出 [固定版本](third_party/versions.json)，初始化 HEonGPU 子模块，并应用 [本地补丁](third_party/patches)。重复执行会检查已有修改，发生冲突时停止，不覆盖本地文件。依赖编译命令见 [第三方说明](third_party/README.md) 和下方构建章节。小型测试记录保留在 `logs/`，历史大型跟踪文件的说明见 [logs/README.md](logs/README.md)。

最新大规模结果见 [1M / 500 簇 / Top-15 / 100 桶 / Top-10 报告](experiment/tfhe_rs/REPORT_1M_C500_U15_B100_K10.md)。该结果不含尚未实现的秘密 BFV gather 和候选转换，不是完整查询耗时。

## 代码结构与职责

```text
src/
├── he/
│   ├── entity/              # Client/Server and TFHE key ownership
│   ├── bfv_distance.*      # Encrypted squared L2
│   ├── tfhe_select.*       # GPU comparisons, secret gather and bucket minima
│   ├── tfhe_topk.*         # Top-k entry point
│   └── detail/             # Selection schedule and internal TFHE state
└── utils/
    ├── types.hpp           # Shared ciphertext and evaluation-key types
    ├── parameters.hpp      # BFV parameters and input bounds
    ├── bfv_crypto.*        # Client-side BFV encryption/decryption
    ├── tfhe_crypto.*       # Client-side TFHE encryption/decryption
    ├── bfv_encoding.hpp    # Plaintext coefficient encoding
    ├── convert.*           # Server-side BFV -> TFHE entry point
    ├── conversion/         # LWE extraction and Rust conversion bridge
    ├── detail/             # Native ABI loader and private key helpers
    └── check_support.hpp   # Test timing and reporting
```

- [Client](src/he/entity/client.hpp) 负责上下文、密钥生成及提供 `evaluation_keys()`；`TfheKeys` 支持只需 TFHE 的独立实验。当前“发送”是同进程传递评估密钥包，尚未实现网络传输。
- [Server](src/he/entity/server.hpp) 复用 BFV 编码器、运算器和 TFHE GPU 评估密钥，提供距离、Top-k、`gather_clusters()`、`random_buckets()` 和 `bucket_minima()` 方法。
- [BfvCrypto](src/utils/bfv_crypto.hpp) 和 [TfheCrypto](src/utils/tfhe_crypto.hpp) 提供加解密，构造一次后复用。工具共享客户端密钥的生命周期；Server 和转换工具只接收评估材料。
- [BfvToTfheConverter](src/utils/convert.hpp) 提供 `convert()`；系数提取、Rust 接口、GPU 转换实现均在 `utils` 中。
- [bfv_distance](src/he/bfv_distance.cpp) 使用 `Enc(||q||²) - 2Σ Enc(q[d])·x[d] + ||x||²`。查询坐标与范数是加密常数多项式，每个输出系数对应一个候选；距离计算无需 relin 或旋转密钥。
- [tfhe_topk](src/he/tfhe_topk.cpp) 复用 PrivateANN 单服务器 Rust 原型的分组排序、裁剪合并网络。组内用 Batcher odd-even merge sorting，组间用裁剪的 bitonic merge。Top-u 调用 `server.topk(records, u)`。
- [tfhe_select](src/he/tfhe_select.cpp) 调用 TFHE-rs GPU **`gt` 和 `if_then_else`**；只比较距离，同一加密条件选择距离和 ID。双输出节点用一次比较、四次字段选择，单输出节点用一次比较、两次字段选择。默认 4 个工作线程执行同层节点。

每条 TFHE 记录分别保存距离和 ID：23 位距离使用 **`FheUint32`**，ID 也使用 **`FheUint32`**，仅距离参与大小比较。无效记录的距离为 `2^score_bits`，大于所有合法值；声明完整 32 位距离时使用 `FheUint64` 保存距离与哨兵，ID 仍为 32 位。公开位宽用于校验输入，同批位宽须一致。

距离相同可任选，排序不保证稳定；同一个加密条件始终选择成对的距离和 ID。内部仍携带距离，完整 ANN 最终只能向客户端返回 ID；`utils::TfheCrypto::decrypt()` 是组件测试的诊断接口。

客户端生成压缩评估密钥，服务器一次性解压为 CUDA key。每个工作线程为整轮 Top-k 设置一次 CUDA key 和独立 stream，并在各层结束处同步；同层各节点读写互不重叠的位置。工作线程异常回传到调用方，退出时清理线程状态。直接构造 `tfhe_rs::Evaluator(key, workers)` 可指定 1–32 个工作线程（1 表示串行），默认 4。库缺失会报错，没有 CPU 或门级后备实现。HEonGPU 内存池初始预留改为可用显存的 5%，上限 25%，为同进程 TFHE-rs 留出显存。单次 100 选 10 的基线和优化实测见 [TFHE-rs 实验](experiment/tfhe_rs/README.md)。

**BFV→原生 TFHE-rs 转换已有实现。** `Client::enable_bfv_tfhe_conversion()` 一次生成跨密钥材料，`utils::BfvToTfheConverter::convert()` 完成系数→torus LWE 提取、密钥切换、bit extraction、模数修正与 radix 组装。默认 GPU 后端对一批距离并行执行 key switching/PBS，各位之间按依赖顺序执行；末尾原生 GPU 整数运算使用 4 个工作线程。GMP/RNS 重构仍在 CPU。转换支持保留完整距离，或显式设置 `drop_bits=7` 复用 CPU 参考的 16 位比较语义。详细精度条件、实验参数和实测见 [BFV_TO_TFHE.md](docs/BFV_TO_TFHE.md)。

OpenFHE 原生 CPU 明密文乘法与原始转换示例放在独立的 [experiment/openfhe](experiment/openfhe/README.md)，不接入上述实体封装。

第 4 步已实现 TFHE 固定扫描：`server.topk(centroids, u)` 的密文输出直接交给 `server.gather_clusters(selected, clusters)`，输出固定 `u*L` 条加密距离与 ID，可继续送入 TFHE Top-k。接口、访问模式和小规模 GPU 验证见 [oblivious select](docs/OBLIVIOUS_SELECT.md)。

桶内最小值使用 `server.bucket_minima(buckets)`，复用现有 `topk_schedule(capacity, 1)`、原生 GPU 距离 `gt`/字段选择和同层执行器。所有桶的同层节点一起调度，输出固定为 `B` 条记录；空桶使用无效密文补齐，返回无效优胜者。后续调用 `server.topk(winners, k)`。桶数与等长容量公开，代码不会根据秘密占用跳过桶。`server.random_buckets(candidates, B, seed)` 对公开候选位置洗牌后均衡分桶，不读取秘密 ID。

单次连续链路可运行 `./build/privateimpl/selection_pipeline_once`：100 条 128D uint8 记录、10 簇选 4 簇、16 桶取 Top-10，完整 23 位距离、20 位 ID，真实 BFV→TFHE 转换，中间无解密。逐阶段计时保存到 `logs/selection_pipeline_split_once.csv`；不含输出净化和网络传输。源码在 `experiment/tfhe_rs/`，`test/` 保持四个文件。

### 阅读 Top-k 和 `tfhe_rs_api`

1. [topk_schedule.hpp](src/he/detail/topk_schedule.hpp)：英文注释说明 CPU 来源、分组宽度、裁剪依据、奇数组数处理和并行层依赖。100 选 10 使用 16 个元素一组，补齐至 112，共 25 层、633 个双输出节点和 96 个只取最小值的节点。
2. [selection_network.hpp](src/he/detail/selection_network.hpp)：执行公开网络，补齐并保留前 k 个结果。
3. [tfhe_select.cpp](src/he/tfhe_select.cpp)：管理 GPU 评估密钥并执行选择，双输出节点调用一次距离 `gt` 和四次字段 `if_then_else`，裁剪节点只生成较小记录，调用一次 `gt` 和两次字段选择；工作线程执行同层节点。
4. [tfhe_rs_api.hpp](src/utils/detail/tfhe_rs_api.hpp) / [tfhe_rs_api.cpp](src/utils/detail/tfhe_rs_api.cpp)：C++ 调用 Rust 库的接口层。声明官方 C ABI 的不透明对象和函数指针，使用 `dlopen/dlsym` 加载 `libtfhe.so`，把 C 返回的错误码转为 C++ 异常。`check.cpp` 对照官方生成的 `tfhe.h` 在编译时核对签名。这一层没有实现比较算法，也没有 BFV→TFHE 转换功能。

当前记录布局统一在 `RecordLayout` 中定义，已移除旧的 packed rank 和 FheUint128 接口。历史基线计时仍保存在实验报告中，不能当作当前分离字段实现的时间。

## 四个测试文件与计时

`test/` 只保留以下四个文件；共享计时与结果输出在 `src/utils/check_support.hpp`。新增簇选择测试复用这些文件。

| 文件 | 检查内容 |
|---|---|
| `check.cpp` | 总入口；检查生成头文件 ABI、动态库、CPU 网络和参数边界，执行三个组件，汇总时间 |
| `bfv_distance_test.cpp` | 100 万条 128 维 uint8 距离；另检查 `(5-3)^2`、三轮三维 L2、128 维最大距离、实体复用及查询不变 |
| `bfc_tfhe_conversion.cpp` | 500 个真实 BFV 距离转换为原生 TFHE-rs，默认截掉低 7 位；验证解密、边界、ID、转换后的 GPU Top-3，以及转换输出驱动的 Top-u→gather→Top-k；支持完整精度检查 |
| `tfhe_topk.cpp` | 原生 GPU Top-k、秘密簇 gather、固定逻辑访问序列；并行桶内最小值→Top-k、空桶/单条/满桶、并列、非 2 次幂容量、32/64 位距离、32 位 ID 与输入不变 |

本机依赖与新测试入口已编译。在能访问 GPU 的终端运行：

```bash
cd /home/student.unimelb.edu.au/xinyutu/PrivateImpl
./build/privateimpl/check
```

结果显示在终端，汇总保存到 `logs/evaluation_times.csv`；可用 `--report PATH` 改路径。`check --cpu` 只运行 CPU 网络和参数检查。三个组件也有同名独立可执行文件：`bfv_distance_test`、`bfc_tfhe_conversion`、`tfhe_topk`。

`check` 返回 0 表示所请求检查全部通过，1 表示失败，2 表示存在未实现/不可运行的组件。转换现在执行真实密文链路，任一结果不匹配会报错。TFHE 秘密簇 gather、随机位置分桶和桶内最小值已接通；完整 ANN 的输出净化尚未完成。独立组件不可运行时返回 77，供 CTest 标为跳过。

### 百万条距离测试

固定规格为 **1 个查询 × 1,000,000 个候选，每个向量 128 维，每个元素为无符号 8-bit（0–255）**。固定种子生成合成数据，按列使用 HEonGPU 编码接口要求的 `uint64_t` 缓冲区。数据分批生成，不在内存中展开全部候选。

每批 4096 条，共 245 批，末批含 576 条真实候选，其余位置补零。客户端只生成一次密钥，并将 128 个查询坐标及 1 个平方范数各加密一次供所有批次复用；服务器对明文候选计算平方 L2。客户端逐条对照整数参考，也校验补齐位置和查询未被修改。补齐位置不计入百万条记录或吞吐量。

默认参数和测试统一使用 `N=4096, t=8380417, max_coordinate=255`。`t` 是 23 位素数，且 `t ≡ 1 (mod 8192)`；这里的 23 位指明文模数位宽，BFV 密文仍使用库的 RNS 表示。最大平方 L2 为 `128 × 255² = 8,323,200 < t`，避免明文模回绕；另有所有坐标差均为 -255 的密文边界测试。

只运行距离测试并查看时间：

```bash
./build/privateimpl/bfv_distance_test
```

### 时间定义

- `evaluation_ms`：服务器方法调用前后同步全部可见 GPU，以 `steady_clock` 测量完成时间；包括方法内的主机调度、编码/传输和 GPU 运算。排除客户端密钥生成、输入加密、结果解密与正确性对照。
- `distance_evaluation_ms`：百万条 128 维距离的 245 批评估时间之和，排除前面的 7 次小规模/边界回归调用。终端及 CSV 单独报告；`distance_records_per_second` 用真实记录数除以该时间计算。
- 总服务器评估时间：统计各组件完成的调用，包括 252 次 BFV 距离调用（7 次回归 + 245 批）、转换组件的源距离/提取诊断/真实转换/Top-k/秘密 gather 小链路，以及 TFHE Top-k/gather/桶内最小值组件的各次评估。百万条工作负载每 16 批显示验证进度，结束时输出总时间；非法输入检查不计入评估。
- `setup_ms`：实体和加解密工具初始化、密钥生成及 GPU server key 初始化；加密、解密另列。TFHE 实体的初始化包含其 BFV 上下文和密钥。
- `wall_ms`：整段测试实际经过的时间，包含初始化、数据准备、评估、验证和对象清理；总入口的值截至 CSV 写入前。
- `conversion_evaluation_ms` 统计真实 BFV→原生 TFHE-rs 转换；源距离和转换后的 Top-3 of 11 单独报告。三者之和仍指原有距离→转换→Top-3 of 11 样例；新增质心 Top-u→gather→候选 Top-k 的三次调用单独打印，计入组件总评估时间。独立提取诊断、客户端操作不计入该和。完整 ANN 尚未实现，时间仍为 **N/A**。组件总和也不是单次 ANN 查询延迟。

百万条工作负载在回归测试后执行，复用已初始化的上下文；未做多次重复统计。500/16 目前只有 CPU 网络正确性检查，没有 GPU 性能结果。本机 L40S-48Q 单次实测：百万条距离服务器评估 **2,683.661 ms**，约 **372,625 条/秒**；含 7 次回归的全部 BFV 评估为 **2,718.328 ms**，组件墙钟 **4,025.566 ms**。系数打包后的全部距离逐条验证通过，原始记录见 [日志](logs/bfv_linear_coefficient_1m.log)。

### 转换测试

```bash
bash third_party/build_bfv_tfhe_bridge.sh
./build/privateimpl/bfc_tfhe_conversion                         # 500 distances, score = distance >> 7
./build/privateimpl/bfc_tfhe_conversion --smoke --full-precision # 11 boundaries, full 23-bit score
```

两条命令是不同的检查配置，无需反复运行同一基准。默认后端是 GPU 批处理；设置 `PRIVATEIMPL_CONVERSION_BACKEND=cpu` 可运行显式 CPU 分解对照。转换密钥须在取得评估密钥包、构造转换工具之前生成。调用示例（`q`、`columns`、`coefficients`、`ids` 为输入数据）：

```cpp
#include "he/entity/client.hpp"
#include "he/entity/server.hpp"
#include "utils/bfv_crypto.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/convert.hpp"

using namespace privateann;
entity::Client client({}, true);
client.enable_bfv_tfhe_conversion();              // CLIENT: generate evaluation keys
utils::BfvCrypto bfv(client);                     // CLIENT: reusable crypto utilities
utils::TfheCrypto tfhe(client.tfhe_keys());
const auto keys = client.evaluation_keys();       // CLIENT -> SERVER: evaluation material
entity::Server server(keys);
utils::BfvToTfheConverter converter(keys);

auto query = bfv.encrypt_bfv_query(q);             // CLIENT: encrypt query and squared norm
auto distance = server.squared_l2(query, columns); // SERVER: encrypted distance
auto records = converter.convert(distance, coefficients, ids, 9, 7); // SERVER: conversion
auto nearest = server.topk(records, 10);          // SERVER: select nearest candidates
auto first = tfhe.decrypt(nearest.front());       // CLIENT: diagnostic verification
```

`drop_bits=7` 时比较 16 位分数，设为 0 保留完整 23 位距离；两者均返回独立的 FheUint32 距离和 ID。截断会增加并列，并列可任选。转换适配库输出已改为双字段包，需重新运行 `build_bfv_tfhe_bridge.sh`；新符号 `bfv_tfhe_convert_split` 会拒绝误用旧库。当前转换参数已做功能验证，尚未给出组合失败概率的形式化证明。

## 依赖状态

TFHE-rs GPU 动态库已安装，ABI 及库加载检查通过。用户 L40S-48Q 不支持 CUDA 内存池；修补 RNGonGPU 和 TFHE-rs 初始化后，已有用户日志确认小规模 BFV 与 TFHE Top-k/Top-u 组件通过。本轮 4096 槽、23 位模数的百万条 128 维 GPU 测试已通过。

首次安装使用 `CARGO_NET_OFFLINE=false TFHE_CUDA_ARCH=89 bash third_party/install_tfhe_rs.sh`，无需 sudo；固定 TFHE-rs 1.5.4、`nightly-2025-11-19`、`high-level-c-api,gpu`。Rust/Cargo 缓存位于 `third_party/rust/`，原生库及头文件位于 `third_party/tfhe-rs-install/`。安装脚本使用 `check.cpp` 的 ABI 专用编译模式核对生成头文件，详见 [第三方依赖说明](third_party/README.md)。

## 重新编译本项目

HEonGPU 已安装在 `third_party/heongpu-install/`。本机命令：

```bash
cmake_bin="$HOME/SecPromptTuning-service/.venv/bin/cmake"
"$cmake_bin" -S . -B build/privateimpl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH="$PWD/third_party/heongpu-install"
bash third_party/build_bfv_tfhe_bridge.sh
"$cmake_bin" --build build/privateimpl -j 4
"$HOME/SecPromptTuning-service/.venv/bin/ctest" --test-dir build/privateimpl --output-on-failure
```

`TFHE_RS_ROOT` 可指定其他安装前缀；运行时 `PRIVATEIMPL_TFHE_RS_LIBRARY` 可指定同版本 GPU `libtfhe.so` 的路径。仅启用 CPU 逻辑构建：

```bash
"$cmake_bin" -S . -B build/logic -DPRIVATEIMPL_BUILD_GPU=OFF
"$cmake_bin" --build build/logic -j 4
"$HOME/SecPromptTuning-service/.venv/bin/ctest" --test-dir build/logic -L cpu --output-on-failure
```

实体默认 BFV 参数为 `N=4096, t=8380417, max_coordinate=255`，支持 128 维 `[0,255]` 的完整距离。129 维最坏情况会溢出，因此接口拒绝；CPU 参数检查和百万条密文验证均已通过。上下文继续使用 HEonGPU 默认 128 位安全级别的系数模数。

## 明文参考

`src/cluster/balanced.py` 提供小规模容量均衡 k-means，保留前 `C*floor(N/C)` 条记录，密集分配内存 O(N²)，最多 4096 条。`experiment/plain_reference.py` 提供 L2、质心选择、随机分桶和 recall@k 参考。

按本轮测试目录整理要求，原 Python 环境检查和明文参考测试已删除；这两个 Python 实现仍保留，未包含在当前四个 HE 测试文件的覆盖范围内。

2026-10-12 调整会话权限后，代理环境已可见 L40S-48Q；此前 GPU 跳过记录对应旧权限环境。秘密簇 gather、公开随机位置分桶和并行桶内最小值已有实现；输出净化和真实数据集端到端实验仍未完成。百万 gather 的容量探针测得每条距离+ID 的原始密文载荷为 **524,544 字节**，1M 输入约 **524.54 GB**；`C=500,u=100` 的 20 万条 gather 输出另需 **104.91 GB**，合计至少 **629.45 GB**，未含密钥、对象和临时工作区。本机无法全量驻留，完整运行时间为 N/A，见 [容量报告](logs/tfhe_gather_1m_u100_split_capacity.json)。
