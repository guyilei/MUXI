# mcDNN RNN 日志解析器

`fix_parse_mcdnn_log.py` 从 MCDNN 日志中定位 RNN 调用，提取可复现 testcase 的参数，并生成可合入 RNN 测试配置的 YAML 片段。

脚本仅使用 Python 标准库，不需要安装额外包。

## 快速开始

在脚本目录执行：

```bash
python3 fix_parse_mcdnn_log.py -i ./run.log -o ./parsed_rnn -s
```

- `-i`：输入日志文件、目录、zip、tar、tar.gz 或 tgz。可重复传入多个来源。
- `-o`：输出目录。
- `-s`：在终端打印每个识别到的 RNN 调用及其解析结果。

Windows PowerShell 中将 `python3` 改为 `python`；多行命令的续行符使用反引号 `` ` ``，Linux shell 使用反斜杠 `\`。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `parsed_data.json` | 每次调用的来源、行号、解析字段、默认值、告警和错误信息，用于追溯。 |
| `shape_definitions.yaml` | `shape_rnn_*` shape 锚点片段。 |
| `rnn_config.yaml` | testcase 片段，引用算法和 shape 锚点。 |

验收时先检查 `parsed_data.json`：`valid_functions` 大于 0，并且目标调用的 `valid` 为 `true`。随后确认两个 YAML 文件非空。

## 支持的 RNN 执行 API

脚本按四类 testcase 语义解析执行 API：

| 类别 | API | 说明 |
| --- | --- | --- |
| 前向 | `mcdnnRNNForwardInference` | 旧式推理前向调用。 |
| 前向 | `mcdnnRNNForwardTraining` | 旧式训练前向调用。 |
| 反向数据 | `mcdnnRNNBackwardData` | 计算输入、隐藏状态等的数据梯度。 |
| 反向权重 | `mcdnnRNNBackwardWeights` | 计算权重梯度。旧名称 `mcdnnRNNBackwardFilter` 也兼容。 |

此外，v8 日志使用通用前向 API `mcdnnRNNForward`，脚本同样会解析，JSON 中的 `mode` 标为 `forward`。

## v6/v8 descriptor 的作用

执行 API 有时只打印 `rnnDesc` 指针，实际参数在前面的配置 API 中。因此脚本会同时解析：

```text
mcdnnSetRNNDescriptor_v6
mcdnnSetRNNDescriptor_v8
mcdnnSetRNNDataDescriptor
```

解析规则是：执行 API 内的字段优先；执行 API 缺失时，使用最近一次 `mcdnnSetRNNDescriptor_v6/v8` 补齐 RNN 参数，使用最近一次 `mcdnnSetRNNDataDescriptor` 补齐序列参数。JSON 中的 `descriptor_api`、`descriptor_version`、`descriptor_source_line` 与 `data_descriptor_source_line` 可用于追溯参数来源。

## 参数解析方法

| testcase 参数 | 日志字段 | 解析方法 |
| --- | --- | --- |
| 序列长度 `T` | `seqLength`、`maxSeqLength`、`seqLengthArray` | 优先读执行 API；v8 缺失时读取 `mcdnnSetRNNDataDescriptor`。`seqLengthArray` 保存为每个样本实际长度。 |
| batch `B` | `x_desc`、`y_desc/dy_desc/dx_desc` 的 `dim:[...]`；或 `batchSize` | 常规前向取 `x_desc` 第 0 维；BackwardData 依次尝试 `y/dy/dx`；v8 没有普通 tensor descriptor 时取 RNNDataDescriptor 的 `batchSize`。 |
| 输入大小 `I` | `inputSize`、`input_size` | 从 v6/v8 RNN descriptor 获取。它是进入 RNN 的向量长度，未必等于 ONNX 模型原始输入的最后一维。 |
| 隐藏大小 `H` | `hiddenSize`、`hidden_size` | 从 RNN descriptor 获取。 |
| 层数 `L` | `numLayers`、`num_layers` | 从 RNN descriptor 获取；它表示单次 RNN API 调用的层数。 |
| 投影大小 `P` | `projSize`、`proj_size` | 从 RNN descriptor 获取。对于未启用 projection 且日志打印 `P=H` 的情况，归一化为 `P=0`。 |
| 算法与结构 | `algo`、`cellMode/cell_mode`、`biasMode/bias_mode`、`dirMode/dir_mode`、`inputMode/input_mode` | 保存 MCDNN 枚举值，用于生成 testcase 名称与算法锚点。 |
| 精度 | `dataType`、`data_type`、`data type` | 支持普通字段和 tensor descriptor 两种写法，并映射为 `fp16`、`tf32`、`bf16` 等。 |
| layout、dropout、clip、wgrad | 对应字段或枚举 | 日志有值则提取；未打印时使用 testcase 默认值，并记录在 `defaulted_fields`。 |

## 当前日志的 v8 示例

若 JSON 中出现如下内容：

```json
"function": "mcdnnRNNForward",
"seq_length": 1,
"batch_size": 1,
"rnn_input_size": 640,
"rnn_hidden_size": 640,
"num_layers": 1,
"descriptor_version": "v8"
```

表示这一次底层 RNN 调用是 `T=1, B=1, I=640, H=640, L=1`。它和模型最外层输入的 shape 不一定相同，因为模型可能先经过 Embedding、索引或其他算子。

## 常见问题

### 未找到可生成 testcase 的执行 API

先搜索：

```bash
grep -nE 'mcdnn(SetRNNDescriptor_v[68]|SetRNNDataDescriptor|RNNForward|RNNBackward)' run.log
```

只出现 `mcdnnCreateRNNDescriptor` 或 `mcdnnCreateRNNDataDescriptor` 时，说明日志只记录了对象创建，尚缺少配置或执行信息，无法生成完整 testcase。

### `valid` 为 false 或 YAML 中没有预期 case

查看：

```bash
rg -n 'parse_errors|缺少必需字段' ./parsed_rnn/parsed_data.json
```

缺少必需字段的调用会保留在 JSON 中，但不会生成对应 testcase。多个参数完全相同的有效调用会去重，只生成一个 logical case。

## 后续扩展

1. 新增执行 API：在 `SUPPORTED_RNN_FUNCTIONS` 增加“函数名: mode”。
2. 新增 RNN 配置 API：在 `RNN_DESCRIPTOR_FUNCTIONS` 增加版本，并在 `_parse_rnn_descriptor()` 中读取新字段。
3. 新增序列 descriptor 格式：在 `RNN_DATA_DESCRIPTOR_FUNCTIONS` 与 `_parse_rnn_data_descriptor()` 中补充字段名。
4. 新增 testcase 参数：先在 `_parse_rnn_call()` 的结果字典中保存，再决定是否加入 `REQUIRED_FIELDS`、shape 锚点和 YAML 生成逻辑。

每次扩展后，至少用一份包含目标 API 的真实日志运行一次，并检查 `parsed_data.json` 中的字段来源和 `parse_errors`。
