# BFV → TFHE-rs 转换

当前已实现真实密文转换，入口为 `Client::enable_bfv_tfhe_conversion()` 和 `utils::BfvToTfheConverter::convert()`。客户端一次生成转换评估材料；服务器执行转换时不持有秘密密钥，不调用解密或重新加密。当前参数固定为 **BFV N=4096、t=8380417**，目标为本项目 TFHE-rs 1.5.4 的 GPU message=4、carry=4 参数。

## 距离输入与编码

客户端把每个 `q[d]` 及 `||q||²` 加密为常数多项式。数据库按系数保存，服务器计算：

```text
Enc(||q||²) * (1 + X + ... + X^(N-1))
  + sum_d Enc(q[d]) * polynomial(-2*x[d][i] mod t)
  + polynomial(||x[i]||²)
```

输出系数 i 是第 i 条记录的距离。最大 128 维 uint8 距离 `8323200 < 8380417`，结果保留完整 23 位。距离阶段仅使用明密文乘法与加法，没有密密文平方、重线性化或槽隔离。

## 转换步骤

| 步骤 | 当前实现 | 执行位置 |
|---|---|---|
| BFV 系数提取 | GMP 重构 RNS，`round(c*2^64/Q)`，按 negacyclic 符号提取 | CPU |
| 跨密钥切换 | 原 BFV 4096 维三元密钥 → TFHE 2048 维二元密钥 | GPU 批量 key switching |
| 高精度消息分解 | TFHE-rs WOPBS bit-extraction 算法的 24 轮调度 | GPU 批量 key switching、PBS、加法与标量乘法 |
| radix 块形成 | 用 PBS 把每个 bit 转为目标 shortint 编码，两 bit 一块 | GPU；公开 Rust 容器组装在 CPU |
| 模数修正、可选截断、组装输出 | 原生 `FheUint64` 乘常数、右移、转为距离 FheUint32；公开 ID 独立 trivial 加密为 FheUint32 | GPU，4 个工作线程和独立 stream |
| Top-k | 原生 TFHE-rs 比较与选择，已有分组裁剪网络 | GPU |

同一轮可同时处理所有选中的距离；一个距离的下一轮依赖前一轮已移除的 bit，因此 24 轮顺序执行。各轮调用库的批处理 CUDA 接口；增加 CPU 线程数不能替代这个批处理。前面的 RNS 重构仍在 CPU，因此当前并非所有转换步骤都在 GPU。

Rust 适配层使用的上游接口包括 `cuda_keyswitch_lwe_ciphertext`、`cuda_programmable_bootstrap_lwe_ciphertext`、`cuda_lwe_ciphertext_*`，以及高层原生整数运算。CPU 对照调用 `extract_bits_from_lwe_ciphertext_mem_optimized`。GPU 中的循环顺序参考上游 `core_crypto/fft_impl/fft64/crypto/wop_pbs/mod.rs::extract_bits`，未自行实现比较器。

## 转换密钥大小

同参数独立探针测得：两把密钥切换密钥和两把自举密钥的展开载荷合计 **1.21219 GiB**，seeded 格式序列化合计 **384.7503 MiB**。另加普通 TFHE 压缩评估密钥 **115.0783 MiB**，传输材料约 **499.829 MiB**。一套材料可复用于全部记录，1M 条不会增大密钥。

探针只验证了 seeded 生成、展开与 GPU 上传，当前转换主链路还未接入 seeded 保存/加载。GPU 运算仍需展开密钥，压缩减少存储和传输量。生产桥接对象同时保留 CPU 与 GPU 密钥副本；详细大小、计量范围及命令见 [测试状态](../TEST_STATUS.md#bfv→tfhe-转换密钥的大小与-seeded-压缩)，[原始 JSON](../logs/bfv_tfhe_key_size.json)。

## 从非二次幂模数恢复距离

提取后的 LWE phase 约为 `m/t`。服务器加公开偏移 `1/(2t)`，然后提取 24 位 torus 分数，得到整数密文：

```text
z ≈ round(2^24 * (m + 1/2) / t)
m = floor(z * t / 2^24)
```

因为 `t < 2^23`，仅量化的半格误差不足以改变最后的整数结果。当全部数值误差满足 `|z - 2^24*(m+1/2)/t| < 1` 时，上式精确恢复 m。实际密文还含 BFV、密钥切换、PBS 和 FFT 误差；**本实现做了边界及随机输入的功能测试，尚未证明组合失败概率**。

转换使用目标 GPU 密钥原有的 GLWE 噪声分布（TUniform 17）、上游 legacy WOPBS 的 769 维辅助 LWE 和噪声分布。跨密钥切换分解为 base_log=8、level=7；提取用 PBS 为 base_log=4、level=12；输出位 PBS 为 base_log=8、level=4。细分解用于维持 23 位消息精度。这是项目的实验组合参数，不能把目标 GPU 默认参数的失败概率直接用于整条转换。

## 截断与 TFHE 类型

- `utils::BfvToTfheConverter::convert(..., id_bits, 0)`：保留完整 23 位距离。
- `utils::BfvToTfheConverter::convert(..., id_bits, 7)`：服务器在密文上计算 `distance >> 7`，最大 `65025`，16 位。
- 两者都返回独立的 **FheUint32 距离 + FheUint32 ID**。ID 是转换输入阶段的公开常量，使用 trivial 加密；后续秘密条件选择产生秘密 ID，不需要解密或重新加密距离。
- 默认组件测试显式使用第二种语义，与 CPU 参考一致；转换工具接口默认保持完整精度。

128 和 255 截断后都变为 1，可任选其一；比较不使用 ID。截断会增加并列，可能改变完整距离的精确 Top-k。测试中的客户端右移只生成参考值，服务器截断发生在 Rust 的原生密文右移中。

适配库使用新符号 `bfv_tfhe_convert_split`。每条输出依次为 `[u64 length][distance bytes][u64 length][ID bytes]`，C++ 端分别导入两个原生密文。修改后必须重建适配库，旧 packed rank 库会因缺少新符号而明确报错。

## 代码位置与复现

- `src/he/bfv_distance.*`、`src/utils/bfv_encoding.hpp`：系数打包明密文距离。
- `src/utils/conversion/bfv_lwe.*`：BFV 系数→torus LWE；保留 CRT 槽隔离用于历史诊断。
- `src/utils/conversion/rust_bridge/bfv_tfhe.rs`：客户端转换密钥生成、CPU 对照、原生整数修正、序列化 C ABI。
- `src/utils/conversion/rust_bridge/gpu.rs`：GPU 批量 bit-extraction 调度。
- `src/utils/convert.*`：接收评估密钥包，校验参数并串联系数提取与原生转换。
- `src/utils/conversion/tfhe_conversion.cpp`：动态加载转换适配层，客户端临时密钥缓冲区清除，原生密文导入和评估密钥身份检查。
- `src/utils/detail/tfhe_rs_api.*`：官方 C API 的声明/动态加载；转换算法在上面的 Rust 适配层中。
- `test/bfc_tfhe_conversion.cpp`：真实距离→原生 TFHE-rs→GPU Top-k 检查；`test/` 仍只有四个文件。

```bash
bash third_party/build_bfv_tfhe_bridge.sh
./build/privateimpl/bfc_tfhe_conversion
./build/privateimpl/bfc_tfhe_conversion --smoke --full-precision
```

构建脚本复用已编译的同版本 TFHE-rs rlib、bincode/rayon 缓存和指定 nightly，把适配库安装到 `third_party/tfhe-rs-install/lib/`。服务器首次加载缺失库会明确报错。`PRIVATEIMPL_CONVERSION_BACKEND=cpu` 选择 CPU 分解对照，不会因为 GPU 失败自动改变后端。

当前同进程测试通过序列化缓冲区导入已形成的原生 TFHE-rs 密文。客户端序列化秘密密钥仅用于生成转换材料的本地初始化调用；缓冲区随后清除，未写入文件，也不会传给 Server。`BridgeEval` 的字段只有评估密钥、CUDA server key 和线程池。

## 历史 packed rank 版本的测试与计时

以下数字来自字段分离前；当前代码的检查结果见 [测试状态](../TEST_STATUS.md)。

11 个完整精度边界值含 `0、1、8323200、127、128、129、65535、65536、255、256、257`。CPU 分解版和 GPU 批量版均通过全部值及转换后原生 GPU Top-3。GPU 版这组转换为 **3589.881 ms**。

CPU/GPU 混合基线的 500 个距离及转换后的 Top-3 全部通过：完整转换 **325815.626 ms**，其中 CPU 密钥切换/分解 **247487.555 ms**、串行提交的 GPU 整数修正 **77913.344 ms**。这是迁移 GPU 批处理前的历史基线。日志：[CPU 基线](../logs/bfv_tfhe_native_500.log)、[GPU 完整精度边界](../logs/bfv_tfhe_gpu_smoke_full.log)。GPU 500 个输入的最新结果见 [测试状态](../TEST_STATUS.md)。

GPU 批处理的 500 个输入已全部通过：转换 **91126.039 ms**，其中批量密钥切换/分解 **41410.221 ms**、4 stream 整数修正及输出 **49317.282 ms**。相对上面的混合基线约快 **3.58 倍**；转换后的原生 GPU Top-3 of 11 也通过，耗时 **1633.693 ms**。这轮使用 `drop_bits=7`，验证 16 位分数和 ID。见 [GPU 500 日志](../logs/bfv_tfhe_gpu_500.log) 和 [计时汇总](../logs/bfv_tfhe_conversion_summary.json)。

服务器计时同步 GPU，包含 RNS 提取、密钥切换、分解、整数修正及输出适配；排除密钥生成、输入加密和客户端验证。单独的 LWE 提取诊断时间不重复计入真实转换。转换测试还报告一批 BFV 距离和转换后 11 选 3 的时间，不能把它标为 500 选 10 或完整 ANN 时间。

## CPU 参考及旧 CRT 问题

参考：`PrivateANN/single-server/he-comparison-exploration/bfv-batch-pipeline`，commit `79389cd5f349bc67a9c4e69945927c00a27f4fc1`。其后端是 MOSFHET/AmortBoot，要求 N=8192，CRT 模式 t=33538049 或系数模式 t=2^23，输出自定义 packed radix。当前移植了系数打包、sample extraction、消息分解的思路，并为 TFHE-rs 的密钥、非二次幂 t 和 radix-4 格式实现适配；不能直接复用旧 `PackedRecords`。

原来的 CRT `(Enc(q)-Plain(x))²` 距离本身正确，但 N=4096/t=8380417 时，乘选择槽的明文 mask 就出错，尚未执行旋转或 LWE 提取。mask 编码后的系数绝对值可达 4189627。只改线性距离、仍用 CRT 打包也失败；采用线性距离和系数打包后，无需 mask/旋转，4096 个距离及 500 个 LWE 全部正确。诊断见 `experiment/bfv_conversion/diagnose.cpp` 和 [日志](../logs/bfv_conversion_diagnose_4096_t23.log)。历史 `--wide-control` 仅留在这个诊断程序。

当前适配器专门接受 BFV N=4096/t=8380417，未改变之前 OpenFHE CKKS→FHEW 输入的互通状态。秘密簇路由已接通；最终 ID 输出净化及真实数据集召回率实验仍未完成。
