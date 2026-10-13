# 第三方依赖

仓库只保存依赖获取脚本、[固定版本](versions.json)、`locks/tfhe-rs.Cargo.lock` 和 [补丁](patches/)。`bash third_party/install.sh` 自动检出固定源码并调用 `apply_patches.py` 恢复 HEonGPU、RNGonGPU、TFHE-rs 的本地修改；下载源码、Rust 工具链和安装产物留在本地。下文“已安装”描述本次测试机器的状态，并不表示二进制文件包含在 Git 仓库中。

运行 `bash third_party/install.sh` 将源码及 HEonGPU 子模块放到本目录。HEonGPU 与 GPU-FFT、GPU-NTT、RNGonGPU 子模块现已编译，库安装在 `third_party/heongpu-install/`。TFHE-rs 原生 GPU C API 库已安装在 `third_party/tfhe-rs-install/`，ABI 与库加载检查通过；净化库仍只有源码。

| 目录 | 上游 | 用途 | 版本策略 |
| --- | --- | --- | --- |
| `heongpu/` | https://github.com/Alisah-Ozcan/HEonGPU | GPU BFV 候选后端 | 默认固定 `versions.json` 中已测试的 commit |
| `tfhe-rs/` | https://github.com/zama-ai/tfhe-rs | GPU TFHE 整数比较/选择候选后端 | 默认 `tfhe-rs-1.5.4`，与参考实验一致；安装后记录 SHA |
| `tfhe-sanitization/` | https://github.com/SNUCP/tfhe-sanitization | 输出净化方案候选 | 默认固定 `versions.json` 中的 commit；兼容性尚未验证 |
| `openfhe/` | https://github.com/openfheorg/openfhe-development | 独立 CPU 明密文乘法实验与原始 CKKS↔FHEW/TFHE 转换示例 | `v1.6.1`，commit `b59e1f291965f14fbee5172cd83a20f5b856204f` |

当前检出 commit：HEonGPU `0d551e4e2db26f2cbec47926e41923dbad18759e`，TFHE-rs `638f4c8bab899659374b98df766ea7eb8acfc55f`（tag `tfhe-rs-1.5.4`），tfhe-sanitization `650a3ebcbad71655ec6e56dd0dfdfa122c128e11`。

HEonGPU 的 CKKS 源文件也需要 NTL，即使本项目只用 BFV。系统安装方式是 Ubuntu 的 `libntl-dev`；无 sudo 时可在联网环境运行 `bash third_party/install_ntl_local.sh`，脚本同时获取运行库 `libntl44` 和 `libgf2x3`，然后配置 HEonGPU 时添加 `-D NTL_ROOT="$PWD/third_party/ntl"`。本项目对 HEonGPU 的 CMake 做了最小修改，以支持 `NTL_ROOT` 并在缺依赖时于配置阶段报错。运行链接到本地 NTL 的程序时，设置 `LD_LIBRARY_PATH="$PWD/third_party/ntl/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"`。

HEonGPU 和 TFHE-rs 没有现成的 BFV→TFHE 转换接口。项目已增加实验适配库，使用原生 GPU key switching/PBS 完成转换；500 个截断距离与转换后 Top-k 已通过，参数与限制见 [转换文档](../docs/BFV_TO_TFHE.md)。净化库基于另一 TFHE 实现，尚未接入；完整单服务器 ANN 协议仍未完成。

转换适配库源码在 `src/utils/conversion/rust_bridge/`。运行 `bash third_party/build_bfv_tfhe_bridge.sh`，复用已编译的 TFHE-rs rlib、Rust 工具链及缓存依赖，安装到 `tfhe-rs-install/lib/libbfv_tfhe_bridge.so`，无需新增依赖下载。修改适配层 Rust 代码后需重新运行此脚本；CMake 构建负责 C++ 部分。

运行 `./build/privateimpl/check` 检查 ABI、原生库、网络逻辑和加密组件，并输出时间。2026-10-12 调整会话权限后，代理环境已可见 L40S-48Q；此前 GPU 跳过记录对应旧权限环境。项目本地 Rust 与 TFHE-rs 构建依赖已安装，Go 尚未安装。

OpenFHE 已安装到 `openfhe-install/`，保持上游源码原样。独立实验直接调用原生接口，不接入现有实体封装；版本、构建命令、百万条 128 维 uint8 明密文乘法的计时定义见 [OpenFHE 实验说明](../experiment/openfhe/README.md)。原始转换示例运行成功不代表已打通 HEonGPU 或 TFHE-rs 的密文格式。

## RNGonGPU 的设备内存分配兼容补丁

RNGonGPU 基线 commit 为 `d9aaa6b5d7cc67a2ca307ef9d6203f25cc2458b4`。本地修改 `heongpu/thirdparty/RNGonGPU/src/lib/rand_aes/aes_rng.cu`：五处临时随机数缓冲区分配统一检查当前设备的 `cudaDevAttrMemoryPoolsSupported`，支持时使用 `cudaMallocAsync`，否则使用 `cudaMalloc`，并直接检查 API 返回值。随机数算法与 GPU 核函数保持原样，释放前仍同步相应 stream。

此补丁针对用户 L40S-48Q 上原第 683 行的 `operation not supported`；用户重跑已确认 `memory pools supported=no`，两项 BFV 测试均通过。TFHE-rs 的分配/释放已有同类检测，但内存池初始化仍缺少检测，另需下述补丁。重新构建 RNGonGPU 补丁：

```bash
# Run from the project root with the existing HEonGPU build configuration.
cmake_bin="$HOME/SecPromptTuning-service/.venv/bin/cmake"
"$cmake_bin" --build build/heongpu --target rngongpu -j 4
"$cmake_bin" --install build/heongpu
"$cmake_bin" --build build/privateimpl -j 4
```

## TFHE-rs 原生 GPU C API

### 新检出后的 HEonGPU 构建

先运行 `bash third_party/install.sh`，并安装 CUDA、CMake ≥ 3.30.4、C++ 编译器、GMP 和 NTL 开发文件。以下架构 89 对应本次 L40S：

```bash
cmake -S third_party/heongpu -B build/heongpu \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_INSTALL_PREFIX="$PWD/third_party/heongpu-install" \
  -DHEonGPU_BUILD_EXAMPLES=OFF -DHEonGPU_BUILD_TESTS=OFF
cmake --build build/heongpu -j 4
cmake --install build/heongpu
CARGO_NET_OFFLINE=false TFHE_CUDA_ARCH=89 bash third_party/install_tfhe_rs.sh
bash third_party/build_bfv_tfhe_bridge.sh
```

本地 NTL 安装还需前述 `NTL_ROOT` 设置。之后按根目录 README 配置和构建本项目。

### 无内存池设备的初始化修复

用户运行在 `cuda/src/device.cu` 原第 76 行的 `cudaDeviceGetDefaultMemPool` 中止。已在 `cuda_setup_mempool` 中逐设备检查 `cudaDevAttrMemoryPoolsSupported`，不支持时跳过内存池设置与预热；分配/释放继续使用 TFHE-rs 已有的 `cudaMalloc/cudaFree` 分支。此修复保留原生 GPU min/max 和密钥参数。能力检测依据见 [NVIDIA 内存分配文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#query-for-support)。

另在 CUDA CMake 中加入可选 `TFHE_CUDA_ARCH` 环境变量，`build.rs` 追踪该变量：无可见 GPU 的构建环境可以显式指定目标架构，避免上游自动退到架构 70。L40S 本机使用 89；不指定时保留上游自动检测行为。构建脚本在 `BUILD_INFO.txt` 记录架构及 `device.cu` 的 SHA-256。

```bash
# Rebuild and install the patched GPU C API using existing dependencies.
CARGO_NET_OFFLINE=true TFHE_CUDA_ARCH=89 bash third_party/install_tfhe_rs.sh
./build/privateimpl/check
```

### 连续显存预分配实验（无原生内存池设备）

用户提供的 [EzPC GPU-MPC gpu_mem.cu](https://github.com/mpc-msri/EzPC/blob/master/GPU-MPC/utils/gpu_mem.cu)
在初始化时用 `cudaMallocAsync/cudaFreeAsync` 预热 40 GiB 的原生 CUDA 内存池，设置高释放阈值，之后从池中反复分配；它明确断言设备支持 memory pools。本机 L40S-48Q 报告不支持，因此不能直接调用该实现。

本地 TFHE 后端新增可选 `PRIVATEIMPL_TFHE_ARENA_MB=8192`：每设备一次普通 `cudaMalloc` 预留 8 GiB 连续空间，后续异步分配路径返回 256 字节对齐切片，并按大小复用切片。释放时记录 CUDA event，跨 stream 复用时等待已记录事件，不依赖默认 stream 或全设备同步；同步 `cuda_drop` 仍保留其同步语义。分配大小与复用只依赖公开元数据。实现为 `cuda/src/utils/fallback_cache.cuh`，通过 `device.cu` 的无内存池分支接入，默认未启用。

这是有明确容量的实验分配器：按大小回收，不做通用空闲区合并；耗尽时报错，不偷偷回退到循环内的 `cudaMalloc`。空间及事件随进程 CUDA context 释放，单次查询结束后保留供后续调用复用。日志分别报告预留、已切分及峰值在用字节，不能把预留的 8 GiB 当作实际算法峰值。密钥参数、加密比较、条件选择和数据访问模式未改变；尚未实现跨记录批量 PBS。

重新运行上方构建脚本即可编译；设置环境变量运行实验，省略变量使用原始分配路径。具体 2/10 桶命令及数据见 [实验说明](../experiment/tfhe_rs/README.md)。

### Rust 与 ABI 构建

上游源码忽略 `Cargo.lock`，因此脚本首次构建会先执行 `cargo generate-lockfile`，之后使用 `cargo build --locked` 复用锁定的依赖版本。请保留生成的 `third_party/tfhe-rs/Cargo.lock`。

运行 `bash third_party/install_tfhe_rs.sh`：固定 1.5.4 源码及其上游 nightly 工具链，启用 `high-level-c-api,gpu`。Rust/Cargo 位于 `third_party/rust/`，安装前缀为 `third_party/tfhe-rs-install/`。脚本在安装前用 `test/check.cpp` 的 ABI 专用编译模式检查生成的 `tfhe.h`，该模式无需 HEonGPU 头文件。无需 sudo；首次构建需要联网、CMake、CUDA 编译器和 libclang。本机已完成构建，版本和 ABI 检查记录见 `tfhe-rs-install/BUILD_INFO.txt`。

Top-k 已移除 HEonGPU TFHE 门级实现，原生整数 min/max 由 TFHE-rs 执行。Client 生成压缩评估密钥，Server 解压为 CUDA key，动态库缺失时明确报错。
