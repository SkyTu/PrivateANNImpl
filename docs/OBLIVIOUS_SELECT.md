# TFHE 秘密簇选择、候选 gather 与桶内最小值

## 已确定的处理链路

```text
BFV 距离 → 转换为 TFHE 记录
  → 质心 Top-u
  → TFHE 秘密 gather（距离和 ID 一起选择）
  → 公开随机位置分桶
  → TFHE 并行桶内最小值
  → 桶优胜者 TFHE Top-k
```

选择及后续比较都使用 TFHE。已实现 `Evaluator` / `Server` 的 `gather_clusters()`、`random_buckets()` 和 `bucket_minima()`，可连续接到现有 Top-k。最终输出净化仍待实现。

## 接口与输入

```cpp
// SERVER: centroid and database records already use the same TFHE key.
auto selected = server.topk(centroid_distances, u);
auto candidates = server.gather_clusters(selected, clusters);
auto buckets = server.random_buckets(candidates, B, public_seed);
auto winners = server.bucket_minima(buckets);
auto nearest = server.topk(winners, k);
// CLIENT: decrypt only after the server computation, for test verification.
```

- `clusters[c][r]` 是公开簇 `c` 的第 `r` 条加密记录，分别保存距离与原始记录 ID；距离哨兵表示无效。
- 质心的 `cluster_id` 必须对应 `clusters` 的公开下标 `0..C-1`。质心 ID 唯一，Top-u 仅按距离选出不同质心，距离相同可任选。
- 所有簇容量相同且非零；当前容量均衡规则使每簇有 `L=N/C` 条保留记录。
- 每个候选记录使用相同的 score/ID 位宽；质心位宽可以与候选不同。距离支持 FheUint32（声明最多 31 位）或 FheUint64（完整 32 位距离加哨兵），ID 独立使用 FheUint32。
- 输出固定为 `u*L` 条加密记录，顺序为 `(质心排名 j, 簇内位置 r)`，可直接用于已有 Top-k。

当前转换接口在秘密选择前为每个 TFHE 距离附上其公开原始 ID。gather 用同一个加密条件分别选择距离和 ID，原始 ID 随距离一起成为秘密输出，无需在选择之后用明文 ID 修补结果。

## 固定扫描与加密条件

对每个输出簇 `j`，先比较其距离是否小于公开无效哨兵 `2^score_bits`，得到加密有效位。将其与加密 ID 相等标志相与，避免补齐值的 ID 为零时误选簇 0。

然后依次扫描全部公开簇 `c`：

```cpp
// SERVER: pseudocode for the fixed public access schedule.
output = encrypted_invalid_records(u, L);
parallel_for_each_public_j(j, u) {
    auto valid = fhe_scalar_lt(selected[j].distance, sentinel);
    for (size_t c = 0; c < C; ++c) {
        auto flag = fhe_and(valid, fhe_scalar_eq(selected[j].id, c));
        for (size_t r = 0; r < L; ++r) {
            auto& out = output[j * L + r];
            out.distance = fhe_if_then_else(flag, clusters[c][r].distance, out.distance);
            out.id = fhe_if_then_else(flag, clusters[c][r].id, out.id);
        }
    }
}
```

- 每个 `(j,c)` 只比较一次，其加密相等标志复用于该簇全部记录。
- 每次记录选择调用两次字段 CMUX，条件相同；有效性随距离哨兵保留。
- 每个公开任务 `j` 独占一个输出区域；最多使用配置的工作线程数，各线程安装独立 CUDA stream，输入密文共享只读。
- `u=1` 只有一个主机任务，其密码运算仍在 GPU。任务数量与分配只由公开 `u` 决定。
- 比较结果始终为密文，没有解密、秘密下标、提前退出或根据秘密标志跳过读取。
- 无效或不存在的加密簇 ID 产生固定长度的无效记录；重复选择标志重复取出相同簇。正常调用使用合法质心的 Top-u 输出。

代码位于 `src/he/tfhe_select.cpp`，调用上游 `scalar_lt`、`scalar_eq`、`fhe_bool_bitand`、`if_then_else`。新增 C API 签名由 `check.cpp` 对照已安装版本的官方 `tfhe.h` 编译校验。

## 访问模式与后续路由

公开信息是 `C,L,u`、数据库簇布局和密文位宽。秘密信息是选中哪些簇、质心排名以及输出位置对应哪个原始记录。

每个查询对每个输出簇扫描全部输入簇，因此逻辑地址序列和输出长度不随选择结果变化。`GatherTrace` 是可选测试诊断：在实际选择位置记录 `(源簇, 源行, 输出位置)`，按公开任务序号保存。测试比较不同秘密选择的完整序列；它不记录各 GPU stream 的物理交错，也不读取密文内容。

`random_buckets(candidates, B, seed)` 已实现公开位置随机分桶。对 gather 的固定 `M=u*L` 个位置使用 `std::mt19937_64(seed)` 和 `std::shuffle`，再将第 i 个位置放入桶 `i % B` 的槽 `i / B`。这与 CPU 原型的“洗牌后轮流分桶”采用相同的均衡规则；PRNG 不同，不声称相同 seed 与 Rust/Python 产生相同排列。当前 C++ 构建下可由种子重现。

桶容量固定为 `ceil(M/B)`，没有丢弃或溢出桶；剩余槽用无效密文补齐。允许 `B>M`，这时公开布局中存在补齐空桶。桶数、容量、种子、位置排列和补齐位置均公开；原始 ID、距离和输入记录的秘密有效性仍保持加密。种子须独立于查询和秘密值，代码只复制不透明密文句柄，不检查密文内容。这段公开路由在 CPU 上完成；加密距离、选择和比较调用 GPU。按秘密 ID 分桶不属于此接口。

均衡随机分桶仍有碰撞损失：一个桶的第二近邻不会进入全局 Top-k。测试用相同公开布局检查每桶及最终距离顺序，允许任选并列记录，并校验 ID 与距离对应且不重复生成；另统计对全部记录穷举 Top-k 的 recall，不能把两种正确性混为一谈。

以上是应用层固定路由的设计和回归检查，不是对 GPU 硬件或密码库所有侧信道的形式化证明。

## 并行桶内最小值

```cpp
// SERVER: buckets have public, equal capacities. Their encrypted contents
// were routed without revealing secret cluster IDs or record IDs.
auto winners = server.bucket_minima(buckets);
auto nearest = server.topk(winners, k);
// CLIENT: decrypt only after server evaluation, for test verification.
```

- 输入为 `B > 0` 个等长桶，每桶公开容量 `1 <= capacity <= 65536`。密文必须使用同一个密钥和距离/ID 位宽。空位置存无效记录；秘密空桶也有完整容量，不能用空 vector 表示。
- 复用 `topk_schedule(capacity, 1)`，每桶需 `capacity-1` 个最小记录选择节点，每个节点调用一次原生距离 `gt`，再用两次 `if_then_else` 选择距离和 ID。并列可任选，无 ID 排序比较，不做完整排序。
- 把不同桶的第 `d` 层节点合并成一个公开层，复用 `execute_layers()` 的工作线程和独立 CUDA stream。桶间及桶内无依赖节点都能并行，层之间同步。总计 `B*(capacity-1)` 次距离比较和两倍数量的字段 CMUX、`ceil(log2(capacity))` 层；容量为 1 时没有比较。
- 固定读取公开网络的全部位置，固定输出 `B` 条。全无效桶的最小值仍为无效；最大合法距离也小于无效距离哨兵。全局 Top-k 会把无效项排在有效项之后；有效桶不足 k 时，输出中保留无效位置，不能在服务器端解密后裁短。
- 调用方必须用公开位置映射或 oblivious 路由形成桶；不能先解密 ID 或占用，再建立不同大小的桶。现有 `random_buckets()` 提供基于公开候选位置的布局。

测试覆盖空桶、单条/满桶、非 2 次幂容量、距离并列、32/64 位距离与 32 位 ID、最大合法值、输入不变和非法布局。另用三个容量为 2 的桶记录 `workers` 和 `peak_inflight`：每桶只有一个比较，因而该例的并发来自不同桶。后者记录重叠的主机 API 调用数，不等同于 GPU 内核并行度或加速比。

## 规模与测试

`N=1,000,000, C=500, L=2000, u=100` 时，当前直接扫描输出 200,000 条记录，需要 50,000 次 ID 相等比较、100 次有效性比较、50,000 次布尔与，以及 **200,000,000 次字段 CMUX**（距离与 ID 各一次）。这些是逻辑调用次数，尚无百万 gather 的运行时间。

真实密文容量探针用同一 TFHE-rs 构建、默认参数分别加密距离和 ID，并验证解密结果。每个 FheUint32 有 16 个 radix 块，每块 2049 个 u64；一对字段载荷为 **524,544 字节**，两者序列化大小之和为 **526,896 字节**。

| 常驻内容 | 原始密文载荷下界 |
|---|---:|
| 1M 条输入距离与 ID | 524.544 GB（488.52 GiB） |
| u=100 对应的 200k 条输出 | 104.909 GB（97.70 GiB） |
| 输入与 gather 输出之和 | 629.453 GB（586.22 GiB） |
| 500 个质心，另计 | 0.262 GB |

GB 为十进制字节单位。下界不含密钥、对象元数据、工作区和后续中间输出；不能用它代表峰值显存。测量时主机可用内存约 242.9 GB，当前全量驻留接口仍无法运行该规模；容量检查返回 `CAPACITY_BLOCKED`（退出码 77），未分配或计算百万条 gather。后续需要压缩存储或按固定公开顺序分批处理。

```bash
bash experiment/tfhe_rs/check_gather_capacity.sh logs/tfhe_gather_1m_u100_split_capacity.json 1000000 500 100 10000 10
```

原始数据见 [双字段容量报告](../logs/tfhe_gather_1m_u100_split_capacity.json)。旧 packed FheUint128 每条 1,049,088 字节的测量保留在 [历史报告](../logs/tfhe_gather_1m_capacity.json)。

### 只先转换质心的顺序

已有连续流水线是“全部 BFV 距离转 TFHE → 质心 Top-u → TFHE gather”。`gather_clusters()` 的候选输入是 `vector<vector<TfheRecord>>`；它不能用 TFHE 选择结果直接读取 BFV 密文。若改为“500 个质心先转换 → Top-15 → 秘密取候选 BFV 距离 → 仅转换 30,000 条”，中间的跨方案 oblivious gather 仍需实现。不能解密簇 ID 后按明文地址取簇，也不能将按明文参考准备的候选输入算作此步骤已完成。

取 `N=1M, C=500, u=15` 后，候选距离/ID 双字段载荷为 **15.73632 GB**，但直接沿用原 TFHE gather 仍需全量 **524.544 GB** 输入，逻辑上有 15,000,000 次记录选择（30,000,000 次字段 CMUX）。减小 u 降低了输出规模，没有消除原 gather 的全量输入要求。

`cluster_bucket_once` 单独测量百万 BFV 距离、500 个真实质心转换及 Top-15，并用公开预制的 30,000 条 TFHE 候选测量 100 桶归约及最终 Top-10。这个入口复用已有算子，明确排除上述 BFV gather 和候选转换；用来判断组件成本，不能声称完整秘密查询已执行。运行方式见 [实验说明](../experiment/tfhe_rs/README.md)。

组件测试仍在既有四个测试文件内：

- `test/tfhe_topk.cpp`：真实加密 Top-u → gather → 候选 Top-k，以及并行桶内最小值 → Top-k；覆盖不同秘密选择的相同逻辑访问序列、并列、簇首尾、`u=1`、`u=C`、不同密文宽度、无效选择/空桶、固定输出容量、输入不变和非法公开元数据。
- `test/bfc_tfhe_conversion.cpp`：复用真实 BFV 距离转换输出，前三个作为质心，其后六个组成三个簇，再执行 Top-2 → gather → Top-3。客户端仅在服务器调用结束后核对原始 ID 与距离。

```bash
./build/privateimpl/tfhe_topk
./build/privateimpl/bfc_tfhe_conversion --smoke --full-precision
```

字段分离后的选择组件日志为 `logs/split_fields_tfhe_component.log`。历史 packed rank 版本的桶内最小值及已有选择回归日志为 `logs/tfhe_bucket_component.log`；此前 gather 和真实转换接入日志为 `logs/tfhe_gather_component.log`、`logs/tfhe_gather_converted.log`。每种测试配置仅执行一轮，不重复大规模基准。具体阶段时间及范围见 [TEST_STATUS.md](../TEST_STATUS.md)。

## 单次连续链路实验

`experiment/tfhe_rs/selection_pipeline_once.cpp` 复用以上接口，执行一次连续的 BFV 距离 → 真实 TFHE 转换 → 质心 Top-u → gather → 随机分桶 → 并行桶内最小值 → 桶优胜者 Top-k。全链路中间没有解密，结束后统一客户端验证。

规格为 `N=100, dimensions=128, C=10, L=10, u=4, B=16, k=10`，uint8 元素，23 位完整距离、20 位原始 ID，分别使用 FheUint32；数据种子 20261012，分桶种子 20261013。预先构造均衡合成簇，质心为各簇数据的量化均值；110 个 BFV 距离全部转换，不直接用明文加密 TFHE 替代转换。40 个候选分到 16 个容量为 3 的桶，8 个公开补齐槽，32 个桶内最小值节点分两层执行。

```bash
cmake --build build/privateimpl --target selection_pipeline_once -j 8
./build/privateimpl/selection_pipeline_once
```

各阶段同步 GPU 后计时；报告桶归约与优胜者 Top-k 的实际工作线程数/同时在途调用数，并保存阶段合计、连续服务器墙钟与包含密钥生成及验证的总墙钟。默认 CSV 为 `logs/selection_pipeline_split_once.csv`。本实验尚不含输出净化、网络传输和真实数据集聚类/召回评估，也不代表百万条已运行。
