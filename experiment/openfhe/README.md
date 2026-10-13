# OpenFHE 原生接口实验

本目录为独立实验，直接调用 OpenFHE，不使用项目的 Client/Server 封装。
`test/` 仍只保留用户指定的四个文件。

## 版本与后端

- OpenFHE `v1.6.1`，commit `b59e1f291965f14fbee5172cd83a20f5b856204f`。
- 源码：`third_party/openfhe/`；本地安装：`third_party/openfhe-install/`。
- Release，GNU C++ 11.4，64-bit native backend，OpenMP，`WITH_NATIVEOPT=ON`。
- 本实验为 **CPU** 运算，执行时设 `OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE`。
- 上游：[仓库](https://github.com/openfheorg/openfhe-development/tree/v1.6.1)、[转换示例](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/scheme-switching.cpp)、[转换说明](https://github.com/openfheorg/openfhe-development/blob/v1.6.1/src/pke/examples/SCHEME_SWITCHING_CAPABILITY.md)。

## 明密文乘法

`ckks_plain_multiply.cpp` 测量一个加密查询与 **1,000,000 个明文向量**的逐元素乘法：

- 查询和每个候选均为 **128 维 uint8（0–255）**，合计 **128,000,000 个乘积**。
- 计算为 `Enc(q[d]) * Plain(x[i][d])`，直接调用 `cc->EvalMult(ciphertext, plaintext)`。
- 完整结果范围为 `0–65025`。CKKS 输入使用 `double` 编码；结果为近似数，解密后四舍五入必须逐项等于整数参考，且绝对误差必须小于 0.5。
- 本实验不累加维度，不计算平方 L2，也不在乘法计时内执行方案转换。
- CKKS 参数：`HEStd_128_classic`、深度 1、`FLEXIBLEAUTO`、first modulus 60 bit、scale 40 bit。环维度由 OpenFHE 选择，本机为 16384；8192 个槽位，每批放 64 个完整向量，百万条共 15625 批。
- 同一个查询重复填入各记录对应的槽位，只加密一次，复用全部批次。明文候选使用固定种子 `0x5a17c3` 分批生成。首条候选全为 255、第二条全为 0；查询包含 0、1、127、128、254、255，覆盖最大乘积及零值。
- 所有乘积逐项解密验证；测试末尾确认查询未被修改。部分批次的补齐槽位也验证为零，但不计入真实乘积数。
- 不生成重线性化密钥或旋转密钥，不增加显式 rescale；只测原生明密文乘法调用。

参数检查中曾发现：在相同深度和 first modulus 下使用 50-bit scale 时，`255*255` 无法正确解密。将 scale 改为 40 bit，给完整 16-bit 乘积留出数值范围；65 条向量的边界及末批补齐检查已通过。这个参数调整没有改变原生乘法实现。

### 运行

从项目根目录执行：

```bash
cmake_bin="$HOME/SecPromptTuning-service/.venv/bin/cmake"
"$cmake_bin" -S experiment/openfhe -B build/openfhe-experiment \
  -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR="$PWD/third_party/openfhe-install/lib/OpenFHE"
"$cmake_bin" --build build/openfhe-experiment -j 4

OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE ./build/openfhe-experiment/ckks_plain_multiply \
  --vectors 1000000 --report logs/openfhe_plain_multiply_1m_128_u8.csv
```

CSV 与终端分别报告：

- `setup_ms`：上下文及公私钥生成。
- `query_encoding_encryption_ms`：客户端查询编码和加密一次。
- `record_encoding_ms`：所有服务器明文批次的编码。
- `multiplication_ms`：全部原生 `EvalMult` 调用时间之和。
- `server_encoding_plus_multiplication_ms`：明文编码与乘法之和。
- `decryption_ms`：客户端结果解密和解码，含最终查询检查的解密。
- `verification_ms`：乘积正确性检查。
- `wall_ms`：从创建上下文到完成校验的墙钟时间，包含数据生成、批次内对象清理及进度输出；不包含 CSV 写入和最终上下文析构。
- `max_absolute_error`：全体输出槽位相对整数参考的最大绝对误差。

这些是一次完整数据量的 CPU 实测；没有显式预热或多轮性能统计。

### 本机结果（2026-10-12）

CPU：Intel Xeon Platinum 8562Y+，32 个虚拟 CPU，8 个 OpenMP 线程。
完整执行 **1,000,000 × 128 = 128,000,000** 个明密文乘积，15625 批全部完成；逐项四舍五入后与整数参考一致，查询未被修改。

| 阶段 | 总时间 |
|---|---:|
| 上下文及密钥生成 | 0.013993 s |
| 查询编码与加密一次 | 0.004448 s |
| 服务器明文编码 | 8.135106 s |
| 原生 `EvalMult` 明密文乘法 | **1.785455 s** |
| 服务器编码 + 乘法 | **9.920561 s** |
| 客户端解密和解码 | 46.461452 s |
| 乘积逐项验证 | 0.521149 s |
| 测试墙钟时间 | **60.405517 s** |

最大绝对误差为 **6.34917014395e-05**，小于整数舍入阈值 0.5。仅按乘法时间计算的吞吐量约 **71,690,411 个标量乘积/秒**，这是 SIMD 批处理的摊销吞吐量。

原始记录：[CSV](../../logs/openfhe_plain_multiply_1m_128_u8.csv)、[输出日志](../../logs/openfhe_plain_multiply_1m_128_u8.log)、[进程时间与内存](../../logs/openfhe_plain_multiply_1m_128_u8.time)。`/usr/bin/time` 的进程墙钟为 60.40 秒，最大 RSS 为 17508 KiB。

## 原始转换示例

保持上游 `scheme-switching.cpp` 源码及示例参数原样，运行：

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE /usr/bin/time -v \
  ./build/openfhe/bin/examples/pke/scheme-switching \
  > logs/openfhe_original_scheme_switching.log \
  2> logs/openfhe_original_scheme_switching.time
```

本机退出码为 0，整体墙钟时间 **21.37 秒**，最大 RSS **2,031,088 KiB**。
该程序会打印转换、函数计算及 min/max 等多个演示的结果，包含密钥生成，且部分示例使用 `TOY` / `HEStd_NotSet` 参数。上述时间既不是单次转换时间，也不是百万条转换时间；示例退出码不等于对所有打印结果做了自动断言。

OpenFHE 的转换目标是自己的 FHEW/TFHE（BinFHE）密文。与 TFHE-rs 的实际互通测试见下一节；上面的乘法实验仅测试 CKKS 明密文乘法。

## 500 个距离：CKKS → OpenFHE LWE → TFHE-rs

**结论：可以导入 TFHE-rs 底层 LWE 并保持原密文的解密结果；目前不能直接接入项目的 TFHE-rs GPU `FheUint128` / `min` / `max`。此外，示例转换参数不足以准确保留本次完整 23 位距离。**

### 复现

```bash
cd /home/student.unimelb.edu.au/xinyutu/PrivateImpl
python3 experiment/openfhe/run_tfhe_interop.py
```

脚本使用已安装的 OpenFHE 和已编译的 TFHE-rs 1.5.4 Rust 库，构建两个独立程序。退出码 **2** 表示桥接未完成，**1** 表示实验出现意外错误；不能将 2 视为完整链路通过。不会运行百万条距离或 GPU Top-k。

- C++：[ckks_tfhe_interop.cpp](ckks_tfhe_interop.cpp)，原生 `EvalCKKStoFHEW`。
- Rust：[tfhe_raw_probe.rs](tfhe_raw_probe.rs)，原生 `LweCiphertext::from_container`、`decrypt_lwe_ciphertext` 和 `is_conformant`。
- 自动构建和运行：[run_tfhe_interop.py](run_tfhe_interop.py)。
- [转换计时 CSV](../../logs/openfhe_ckks_tfhe_500.csv)、[转换日志](../../logs/openfhe_ckks_tfhe_500.log)、[TFHE-rs 导入日志](../../logs/tfhe_rs_openfhe_import_500.log)、[汇总 JSON](../../logs/openfhe_tfhe_interop_summary.json)。

### 输入和执行边界

用明文计算 **500 个 128 维 uint8 质心**与一个查询的平方 L2。固定数据种子，覆盖距离 0、1、并列值、最大值 **8,323,200** 和随机质心。距离需要 **23 位**，即使原始坐标只有 8 位。

500 个完整距离编码进一个 CKKS 密文，使用 512 槽位，末尾 12 槽补零；每次只提取 500 个 LWE 密文。CKKS 使用 `HEStd_128_classic`、N=16384、深度 3、first modulus 60、scale 50、`FLEXIBLEAUTOEXT`；BinFHE 使用 `STD128`、n=1305。保留上游示例的 LWE 密文模数 `q=2^25`、中间转换模数 27 位。

密钥生成、加密、服务器转换、客户端验证均在英文注释中标明。服务器转换和系数导入不调用解密。客户端诊断使用同一 OpenFHE 私钥验证导入后的 LWE 相位，**没有解密后重新加密，也没有直接用明文创建 TFHE-rs 密文**。含合成测试私钥的跨进程诊断文件只保存在权限受限的临时目录，脚本结束时自动删除，不写入日志。

### 两个独立检查

同一个 CKKS 输入执行两次转换，每次 500 个元素；第二次是定位精度问题的对照，不是另外一批质心。

| 检查 | 明文模数 p | 最近一次实测结果 |
|---|---:|---|
| `full23`：保留完整距离，主动测试超出推荐噪声余量的参数 | 8,388,608 | **65/500** 完全正确；最大循环整数误差 10，失败 |
| `modular_control`：示例的 `q/(2*beta)`，beta=128 | 131,072 | **500/500** 等于原距离 **mod p**；仅 3/500 等于完整距离 |
| 导入 TFHE-rs 原始 LWE | 两个参数分别测试 | 各 **500/500** 与 OpenFHE 解密结果一致 |
| TFHE-rs 默认 GPU 参数兼容性 | 两个参数分别测试 | 各 **0/500** 符合原始 LWE 维度要求；各 **0/500** 符合 `FheUint128` 要求 |

导入成功说明模数提升及系数布局正确，不能修复转换前已丢失的精度。推荐范围内的模结果不能作为原距离排序，因为取模不保持大小关系。密钥与加密随机，因此重新运行时 `full23` 的错误数量会变化；先行运行同样失败（62/500 正确）。

### 时间（CPU，8 个 OpenMP 线程，2026-10-12）

| 阶段 | 最近一次时间 |
|---|---:|
| 明文计算 500 个距离（包含测试数据生成） | 0.110 ms |
| 上下文、密钥及转换密钥生成，共用一次 | 212.909 ms |
| CKKS 编码及加密，共用一次 | 6.606 ms |
| `full23` 预计算 | 150.490 ms |
| `full23` 转换 500 个元素 | 174.000 ms，**精度检查失败** |
| `modular_control` 预计算 | 104.113 ms |
| `modular_control` 转换 500 个元素 | 169.170 ms，模意义检查通过 |
| TFHE-rs 系数提升及底层对象构造 | 每批约 0.11–0.12 ms |
| 两组实验总墙钟，含验证及进程启动，排除编译 | 951.665 ms |

计时来自一次执行，无重复统计。预计算可在相同参数下复用；TFHE-rs 导入计时不含文件读取、客户端解密、GPU 上传或计算。**有效的完整距离转换时间、GPU 比较时间和完整链路总评估时间尚不可用**，不能将失败尝试的 174 ms 当作可用方案性能。

### 接入 GPU 比较的具体障碍

1. **密文维度和密钥不同。** OpenFHE 输出 n=1305、三元私钥；现有 TFHE-rs GPU 参数的输入 LWE 维度是 2048，内部小 LWE 维度是 920，使用自己的二元私钥。把相同系数装入 Rust 对象不会产生对应的 TFHE-rs evaluation key；需要匹配的密钥切换/初始化方案。
2. **整数表示不同。** OpenFHE 每个距离是一个大明文空间的 LWE。项目的 `FheUint128` 需要 **64 个 radix 块**，每块 message=4、carry=4；23 位距离本身也需要 12 个有效的 2-bit 数位。需要在密文上分解数位并转换编码，不能把一个 LWE 直接当成这些数位。[TFHE-rs shortint 官方说明](https://docs.zama.org/tfhe-rs/references/fine-grained-apis/shortint)。
3. **提升密文模数还不等于转换消息编码。** 实验将 `(a,b)` 各乘 `2^(64-25)`，保持 `b-a·s` 的相位；`full23` 的消息步长为 `2^41`，GPU shortint 的消息步长为 `2^59`，后者还预留进位和 padding 空间。直接更改 metadata 不会改变真实密文编码。
4. **完整距离精度。** 当前 q=2^25，示例建议 p=q/256=2^17。若按相同余量容纳 p=2^23，则需要 q=2^31。对原生 `GenerateBinFHEContext(STD128, false, 31)` 的实际调用报错：`logQ > 29 is not supported`。这是此版本这条参数生成路径的限制；不代表所有 OpenFHE 参数方案都不可能实现 23 位。中间转换模数另有安全约束，本实验没有通过任意增大它来声称问题解决。

Rust 已实际执行原生 conformance 检查，而不是仅根据两个库的序列化格式推断不兼容。检查失败后没有调用 GPU `min/max`，也没有生成一个假通过的 Top-k 结果。完整接入还需要实现并验证上述密文桥接；本实验没有证明其理论上不可实现。

源码依据：OpenFHE `src/binfhe/lib/lwe-pke.cpp`、`src/binfhe/lib/binfhecontext.cpp`、`src/pke/include/scheme/scheme-swch-params.h`；TFHE-rs `tfhe/src/high_level_api/keys/inner.rs`、`tfhe/src/shortint/parameters/multi_bit.rs`、`tfhe/src/shortint/encoding.rs`、`tfhe/src/shortint/client_key/atomic_pattern/standard.rs`。

## CPU 原生 max 与 100 选 10

`ckks_topk_cpu.cpp` 直接使用 OpenFHE 的 `EvalMaxSchemeSwitching`、`EvalMinSchemeSwitching` 和 `EvalCompareSchemeSwitching`。这些接口接收并返回 CKKS 密文，在内部通过 FHEW/TFHE `EvalSign` 判断差值符号。它们属于 CKKS↔FHEW 混合接口，不能当作直接接受单个 BinFHE `LWECiphertext` 的整数 max 函数。当前 OpenFHE v1.6.1 原生接口中没有找到 Top-k。

本实验的 Top-k 是 10 轮原生 argmin 的组合，每轮使用加密 one-hot 标记排除选中的候选。标记通过原生比较 `0.5 < indicator` 刷新回 CKKS，随后累加到加密排除掩码，下一轮输入为 `original + 256 * excluded`。中间结果不解密；全部轮次结束后才执行客户端校验。没有自行实现逐位比较电路，也不使用 TFHE-rs。

### 输入与参数

- 100 个随机生成的 128 维 uint8 质心，一个查询；包含距离 0 和最大距离 8,323,200。明文计算距离后，加密距离用于本实验。
- 目标为 **最小的 10 个距离及其位置**，按升序逐项验证位置。输入没有并列距离，最小相邻距离间隔为 150，第 10/11 名间隔为 29,288。并列或间隔更小的输入不在此次正确性结论内。
- 距离除以 `2^16` 后用 CKKS 编码，保留小数，**不做整数截断**；原生接口要求参与选择的数量是 2 的幂，因此补齐到 128。
- min 的补齐值为 256；max 的补齐值为 -256。加密排除的惩罚为 256。
- CKKS：`HEStd_128_classic`、自动 N=65536、深度 23、`FLEXIBLEAUTOEXT`、scale 50、first modulus 60、`NumLargeDigits=4`。
- FHEW：`STD128`、n=1305、q=2^25、中间模数 27 位；比较预计算 p=131072、scaleSign=128。当前输入和排除值对应的差值保持在有符号比较区间内。
- CPU：Intel Xeon Platinum 8562Y+，使用 8 个 OpenMP 线程。没有 GPU 计算。

这里要求 Top-k 的位置全部正确。CKKS 返回的数值仍是近似值，另行报告选中距离的最大数值误差，不将其描述为完整 23 位精确整数输出。本实验依靠所测数据的距离间隔进行排名，不能代替上一节完整整数转换的精度测试。

### 运行与计时边界

```bash
cd /home/student.unimelb.edu.au/xinyutu/PrivateImpl
cmake_bin="$HOME/SecPromptTuning-service/.venv/bin/cmake"
"$cmake_bin" -S experiment/openfhe -B build/openfhe-experiment \
  -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR="$PWD/third_party/openfhe-install/lib/OpenFHE"
"$cmake_bin" --build build/openfhe-experiment --target ckks_topk_cpu -j8
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE /usr/bin/time -v \
  ./build/openfhe-experiment/ckks_topk_cpu logs/openfhe_topk_cpu_100_10.csv \
  > logs/openfhe_topk_cpu_100_10.log 2> logs/openfhe_topk_cpu_100_10.time
```

- `native_max`：100 个真实输入补齐到 128 后，一次原生 max/argmax 的评估时间，独立于 Top-k。
- `native_argmin`：每轮原生 min/argmin 的完整评估时间，含内部的方案转换。
- `encrypted_mask_refresh_update`：前 9 轮的标记刷新、累加和候选更新；末轮为 0。
- `topk_evaluation_total`：10 次 argmin + 9 次掩码更新，不含密钥生成、初始预计算、输入加密、独立 max 测试以及最终解密校验。
- setup、预计算、加密和验证单独记录。这是一个逐轮 argmin 基线，不代表最快的 Top-k 实现。
- 参数探索中停止了一次深度 25、自动 N=131072 的预跑；记录保存在 `logs/openfhe_topk_cpu_depth25_interrupted.*`，不能当作成功测量。

原始记录：[CSV](../../logs/openfhe_topk_cpu_100_10.csv)、[输出日志](../../logs/openfhe_topk_cpu_100_10.log)、[进程时间与内存](../../logs/openfhe_topk_cpu_100_10.time)。只有程序退出 0、最终日志出现全部位置通过的记录，才能将相应计时作为通过正确性检查的结果。

### 本机实测结果（2026-10-12）

本次退出码 **0**。argmax 的位置以及最小的 10 个位置按顺序全部验证通过，补齐项未被选中，原始输入未被修改。Top-10 索引为 `[0, 28, 49, 9, 48, 36, 26, 85, 32, 61]`。这是一次完整运行，未做多次重复统计。

| 阶段 | 实测时间 |
|---|---:|
| 上下文、密钥及转换/自举密钥生成 | 5.839835 s |
| 比较预计算 | 0.156061 s |
| 客户端编码和加密（包含独立 max 输入） | 0.277943 s |
| 独立原生 max/argmax 调用 | **69.715937 s** |
| Top-k 的 10 次 argmin | 691.538304 s |
| 前 9 轮加密掩码刷新和更新 | 370.008102 s |
| **100 选 10 服务器评估总时间** | **1061.546406 s ≈ 17 分 41.55 秒** |
| 客户端解密与验证 | 0.371727 s |
| 程序墙钟（包含独立 max、初始化与验证） | 1137.991270 s |

`/usr/bin/time` 记录进程总时间 18 分 58.37 秒、最大 RSS 12,193,584 KiB（约 11.63 GiB）、退出码 0。表中 Top-k 时间不包含独立 max，不能将它们加起来作为一次 Top-k 查询时间。

**正确性范围：位置排序通过，返回的距离数值不满足精确整数要求。** 返回距离的最大绝对误差为 **9386.357146**。例如最小距离的参考值为 0，返回 CKKS 数值还原后为 2526.198113；第 9 名参考值为 2445099，返回值为 2454485.357146。因此不能把本实验描述为“完整 23 位距离计算全部正确”，也没有对独立 max 返回的标量数值做精确整数断言。若需要准确距离，应保留原始距离或另行改进数值精度；这次测试结论仅覆盖上述固定数据的 argmax/Top-k 位置。

本次接口在内部执行方案转换和近似 CKKS 选择，因此该时间不能等同于“纯 BinFHE/TFHE Top-k 的耗时”。它也不能据此与 TFHE-rs 的不同数据宽度、网络或 GPU 参数直接做性能比较。机器可读汇总见 [JSON](../../logs/openfhe_topk_cpu_100_10_summary.json)。

## 从源码构建 OpenFHE

```bash
git clone --depth 1 --branch v1.6.1 --recurse-submodules --shallow-submodules \
  https://github.com/openfheorg/openfhe-development.git third_party/openfhe
cmake_bin="$HOME/SecPromptTuning-service/.venv/bin/cmake"
"$cmake_bin" -S third_party/openfhe -B build/openfhe \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/third_party/openfhe-install" \
  -DBUILD_UNITTESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=ON \
  -DBUILD_EXTRAS=OFF -DWITH_OPENMP=ON -DWITH_NATIVEOPT=ON
"$cmake_bin" --build build/openfhe --target scheme-switching -j 16
"$cmake_bin" --install build/openfhe
```
