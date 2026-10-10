#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从包含任意应用输出的 MCDNN 日志中提取 RNN testcase。

输出：
  - parsed_data.json：带来源与校验信息的中间结果
  - shape_definitions.yaml：RNN shape anchor
  - rnn_config.yaml：RNN testcase

输入可以是普通日志、目录、zip，或者 tar/tar.gz/tgz 压缩包。压缩包在内存中读取，
不会把成员解压到文件系统。
"""

import argparse
import json
import re
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FUNCTION_HEADER_RE = re.compile(
    r"^[^\n]*?\[MCDNN\]\[INFO\]\s+function\s+"
    # 某些收集工具会丢失一个括号，日志中会变成 "Descriptor( called:"。
    # 函数名仍然完整，故此处宽松识别括号，避免漏掉后面的 RNN API。
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(?\s*\)?\s+called:\s*$",
    re.MULTILINE,
)

# 四个需要生成 testcase 的执行 API。BackwardFilter 是历史别名，统一归为
# BackwardWeights，保留它是为了兼容旧日志。
SUPPORTED_RNN_FUNCTIONS = {
    # v8 的通用前向 API；它不在函数名中区分 inference/training。
    "mcdnnRNNForward": "forward",
    "mcdnnRNNForwardInference": "inference",
    "mcdnnRNNForwardTraining": "training",
    "mcdnnRNNBackwardData": "backward_data",
    "mcdnnRNNBackwardWeights": "backward_weights",
    "mcdnnRNNBackwardFilter": "backward_weights",
}

# v6/v8 不直接执行 RNN；它们负责把 RNN 配置写进 rnnDesc。后面的
# Forward/Backward 调用可能只打印 rnnDesc 句柄，因此需要保存最近一次描述符配置并补全它。
RNN_DESCRIPTOR_FUNCTIONS = {
    "mcdnnSetRNNDescriptor_v6": "v6",
    "mcdnnSetRNNDescriptor_v8": "v8",
}

# v8 使用单独的 RNNDataDescriptor 记录 T、B、layout 和 sequence lengths。
RNN_DATA_DESCRIPTOR_FUNCTIONS = {"mcdnnSetRNNDataDescriptor": "v8"}

REQUIRED_FIELDS = (
    "seq_length",
    "algo",
    "cell_mode",
    "bias_mode",
    "dir_mode",
    "input_mode",
    "data_type_enum",
    "num_layers",
    "batch_size",
    "input_size",
    "hidden_size",
)


def _decode_log(data: bytes) -> str:
    """尽量无损地读取日志；非法 UTF-8 字节不会阻断整个批处理。"""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("gb18030")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


def _looks_like_text(data: bytes) -> bool:
    return b"\x00" not in data[:4096]


def iter_log_sources(paths: Sequence[str]) -> Iterable[Tuple[str, str]]:
    """遍历普通文件、目录和压缩包，返回 (来源名, 文本)。"""
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在：{path}")

        if path.is_dir():
            # 避免用户把工作目录作为输入时，又把脚本缓存和上次生成的输出读一遍。
            ignored_parts = {".git", "__pycache__", "parsed_rnn", "output", "sample_out"}
            for child in sorted(
                p for p in path.rglob("*")
                if p.is_file() and not any(part in ignored_parts for part in p.parts)
            ):
                yield from _read_one_path(child)
        else:
            yield from _read_one_path(path)


def _read_one_path(path: Path) -> Iterable[Tuple[str, str]]:
    # is_zipfile/is_tarfile 都会打开文件探测格式。普通日志不必承担两次额外 I/O。
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                data = archive.read(info)
                if _looks_like_text(data):
                    yield f"{path}!{info.filename}", _decode_log(data)
        return

    if suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")) and tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            members = sorted(
                (member for member in archive.getmembers() if member.isfile()),
                key=lambda member: member.name,
            )
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                if _looks_like_text(data):
                    yield f"{path}!{member.name}", _decode_log(data)
        return

    data = path.read_bytes()
    if _looks_like_text(data):
        yield str(path), _decode_log(data)


def _find_value(text: str, names: Sequence[str]) -> Optional[str]:
    alternatives = "|".join(re.escape(name) for name in names)
    match = re.search(
        rf"(?:^|[,:{{\s])(?:{alternatives})\s*[:=]\s*([^,}}\s]+)",
        text,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _find_last_value(text: str, names: Sequence[str]) -> Optional[str]:
    """返回字段最后一次出现的值。

    同一个 descriptor 的摘要行和详细行有时会重复打印同一字段；详细行通常在后面，
    因此用最后一次值覆盖摘要中的旧值。
    """
    alternatives = "|".join(re.escape(name) for name in names)
    matches = list(
        re.finditer(
            rf"(?:^|[,:{{\s])(?:{alternatives})\s*[:=]\s*([^,}}\s]+)",
            text,
            re.MULTILINE,
        )
    )
    return matches[-1].group(1).strip() if matches else None


def _find_data_type_enum(text: str) -> Optional[str]:
    # 某些日志写 data_type/dataType，tensor descriptor 则常写 "data type"。
    return _clean_enum(_find_last_value(text, ("data_type", "dataType", "data type")))


def _find_int(text: str, names: Sequence[str]) -> Optional[int]:
    value = _find_value(text, names)
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def _find_float(text: str, names: Sequence[str]) -> Optional[float]:
    value = _find_value(text, names)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _find_dims(text: str, descriptor_names: Sequence[str]) -> Optional[List[int]]:
    alternatives = "|".join(re.escape(name) for name in descriptor_names)
    match = re.search(
        # descriptor 名必须是完整字段，不能让 xDesc 命中 hxDesc/dhxDesc，
        # 也不能让 y_desc 命中 dy_desc。
        rf"(?<![A-Za-z0-9_])(?:{alternatives})\s*:[^\n]*?dim\s*:\s*\[([^\]]+)\]",
        text,
    )
    if not match:
        return None
    try:
        return [int(item.strip()) for item in match.group(1).split(",")]
    except ValueError:
        return None


def _find_sequence_lengths(text: str) -> Optional[List[int]]:
    match = re.search(
        r"(?:seqLengthArray|seq_length_array|devSeqLengths|dev_seq_lengths)"
        r"\s*[:=]\s*\[([^\]]*)\]",
        text,
    )
    if not match:
        return None
    try:
        return [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
    except ValueError:
        return None


def _clean_enum(value: Optional[str]) -> Optional[str]:
    return value.rstrip(",") if value else None


def _find_layout(text: str) -> Optional[str]:
    match = re.search(r"MCDNN_RNN_DATA_LAYOUT_[A-Z_]+", text)
    if match:
        return match.group(0)
    return _find_value(text, ("layout", "data_layout", "dataLayout"))


def _parse_layout(text: str) -> str:
    return _find_layout(text) or "MCDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED"


def _find_wgrad_mode(text: str) -> Optional[str]:
    return _find_value(
        text,
        ("wgrad_mode", "weight_grad_mode", "weightGradMode", "dw_mode", "dwMode"),
    )


def _parse_wgrad_mode(text: str) -> str:
    return _find_wgrad_mode(text) or "MCDNN_RNN_WGRAD_MODE_ADD"


def precision_name(data_type_enum: Optional[str]) -> Optional[str]:
    mapping = {
        "MCDNN_DATA_HALF": "fp16",
        "MCDNN_DATA_FLOAT": "tf32",
        "MCDNN_DATA_DOUBLE": "fp64",
        "MCDNN_DATA_BFLOAT16": "bf16",
        "MCDNN_DATA_INT8": "int8",
    }
    return mapping.get(data_type_enum) if data_type_enum else None


class MCDNNRNNParser:
    """MCDNN RNN 调用解析器。"""

    def parse_sources(self, sources: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
        functions: List[Dict[str, Any]] = []
        descriptors: List[Dict[str, Any]] = []
        source_summaries: List[Dict[str, Any]] = []
        seen_api_names: List[str] = []

        for source_name, content in sources:
            parsed, source_descriptors, source_api_names = self._parse_source(
                source_name, content
            )
            functions.extend(parsed)
            descriptors.extend(source_descriptors)
            seen_api_names.extend(source_api_names)
            source_summaries.append(
                {
                    "source": source_name,
                    "rnn_function_count": len(parsed),
                    "valid_rnn_function_count": sum(item["valid"] for item in parsed),
                    "rnn_descriptor_count": len(source_descriptors),
                }
            )

        errors = [
            {
                "source": item["source"],
                "source_line": item["source_line"],
                "function": item["function"],
                "errors": item["parse_errors"],
            }
            for item in functions
            if not item["valid"]
        ]

        return {
            "total_sources": len(source_summaries),
            "total_functions": len(functions),
            "valid_functions": sum(item["valid"] for item in functions),
            "functions": functions,
            "rnn_descriptors": descriptors,
            "seen_rnn_related_apis": sorted(
                {name for name in seen_api_names if "RNN" in name or "Rnn" in name}
            ),
            "parse_errors": errors,
            "sources": source_summaries,
            "timestamp": datetime.now().isoformat(),
        }

    def parse(self, log_content: str) -> Dict[str, Any]:
        """兼容旧调用方式。"""
        return self.parse_sources((("<memory>", log_content),))

    def _parse_source(
        self, source_name: str, content: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        headers = list(FUNCTION_HEADER_RE.finditer(content))
        results: List[Dict[str, Any]] = []
        descriptors: List[Dict[str, Any]] = []
        source_api_names: List[str] = []
        latest_descriptor: Optional[Dict[str, Any]] = None
        latest_data_descriptor: Optional[Dict[str, Any]] = None

        for index, header in enumerate(headers):
            function_name = header.group("name")
            if "RNN" in function_name or "Rnn" in function_name:
                source_api_names.append(function_name)
            # 绝大部分日志 header 是卷积、Tensor descriptor 等，与 RNN 无关。
            # 先过滤再切片、再计算行号，避免大日志中出现 O(调用数×文件大小) 的扫描。
            is_rnn_descriptor = function_name in RNN_DESCRIPTOR_FUNCTIONS
            is_rnn_data_descriptor = function_name in RNN_DATA_DESCRIPTOR_FUNCTIONS
            is_rnn_execution = function_name in SUPPORTED_RNN_FUNCTIONS
            if not (is_rnn_descriptor or is_rnn_data_descriptor or is_rnn_execution):
                continue
            end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
            segment = content[header.start():end]
            source_line = content.count("\n", 0, header.start()) + 1
            if is_rnn_descriptor:
                latest_descriptor = self._parse_rnn_descriptor(
                    source_name, source_line, function_name, segment
                )
                descriptors.append(latest_descriptor)
                continue
            if is_rnn_data_descriptor:
                latest_data_descriptor = self._parse_rnn_data_descriptor(
                    source_name, source_line, function_name, segment
                )
                continue
            results.append(
                self._parse_rnn_call(
                    source_name,
                    source_line,
                    function_name,
                    segment,
                    latest_descriptor,
                    latest_data_descriptor,
                )
            )

        return results, descriptors, source_api_names

    def _parse_rnn_descriptor(
        self,
        source_name: str,
        source_line: int,
        function_name: str,
        text: str,
    ) -> Dict[str, Any]:
        """解析 v6/v8 RNN descriptor，供紧随其后的执行 API 补全字段。"""
        return {
            "source": source_name,
            "source_line": source_line,
            "function": function_name,
            "version": RNN_DESCRIPTOR_FUNCTIONS[function_name],
            "algo": _clean_enum(_find_last_value(text, ("algo", "algorithm"))),
            "cell_mode": _clean_enum(_find_last_value(text, ("cell_mode", "cellMode"))),
            "bias_mode": _clean_enum(_find_last_value(text, ("bias_mode", "biasMode"))),
            "dir_mode": _clean_enum(_find_last_value(text, ("dir_mode", "dirMode", "direction"))),
            "input_mode": _clean_enum(_find_last_value(text, ("input_mode", "inputMode"))),
            "clip_mode": _clean_enum(_find_last_value(text, ("clip_mode", "clipMode"))),
            "clip_nan_opt": _clean_enum(_find_last_value(text, ("clip_nan_opt", "clipNanOpt"))),
            "data_type_enum": _find_data_type_enum(text),
            "math_prec": _clean_enum(_find_last_value(text, ("math_prec", "mathPrec"))),
            "math_type": _clean_enum(_find_last_value(text, ("math_type", "mathType"))),
            "rnn_input_size": _find_int(text, ("input_size", "inputSize")),
            "rnn_hidden_size": _find_int(text, ("hidden_size", "hiddenSize")),
            "logged_proj_size": _find_int(text, ("proj_size", "projSize")),
            "num_layers": _find_int(text, ("num_layers", "numLayers")),
            "dropout": _find_float(text, ("drop_rate", "dropout", "dropout_rate")),
            "lclip": _find_float(text, ("lclip", "left_clip")),
            "rclip": _find_float(text, ("rclip", "right_clip")),
            "layout": _find_layout(text),
            "wgrad_mode": _find_wgrad_mode(text),
        }

    def _parse_rnn_data_descriptor(
        self,
        source_name: str,
        source_line: int,
        function_name: str,
        text: str,
    ) -> Dict[str, Any]:
        """解析 v8 RNNDataDescriptor 中的序列和 batch 信息。"""
        return {
            "source": source_name,
            "source_line": source_line,
            "function": function_name,
            "version": RNN_DATA_DESCRIPTOR_FUNCTIONS[function_name],
            "seq_length": _find_int(
                text, ("maxSeqLength", "max_seq_length", "seqLength", "seq_length")
            ),
            "batch_size": _find_int(text, ("batchSize", "batch_size")),
            "vector_size": _find_int(text, ("vectorSize", "vector_size")),
            "data_type_enum": _find_data_type_enum(text),
            "layout": _find_layout(text),
            "sequence_lengths": _find_sequence_lengths(text),
        }

    def _parse_rnn_call(
        self,
        source_name: str,
        source_line: int,
        function_name: str,
        text: str,
        descriptor: Optional[Dict[str, Any]] = None,
        data_descriptor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        x_dims = _find_dims(text, ("x_desc", "xDesc"))
        dx_dims = _find_dims(text, ("dx_desc", "dxDesc"))
        dy_dims = _find_dims(text, ("dy_desc", "dyDesc"))
        hx_dims = _find_dims(text, ("hxDesc", "hx_desc"))
        y_dims = _find_dims(text, ("y_desc", "yDesc"))
        w_dims = _find_dims(text, ("wDesc", "w_desc"))
        dw_dims = _find_dims(text, ("dwDesc", "dw_desc"))

        # Forward/BackwardWeights 都有 x descriptor；BackwardData 没有 x，
        # 它的 batch 必须从 y/dy/dx 的第一维获取。
        batch_dim_sources: List[Tuple[str, List[int]]] = []
        if function_name == "mcdnnRNNBackwardData":
            batch_dim_sources = [
                (name, dims)
                for name, dims in (
                    ("y_desc", y_dims),
                    ("dy_desc", dy_dims),
                    ("dx_desc", dx_dims),
                )
                if dims
            ]
        elif x_dims:
            batch_dim_sources = [("x_desc", x_dims)]

        batch_dims_source = batch_dim_sources[0][0] if batch_dim_sources else None
        batch_dims = batch_dim_sources[0][1] if batch_dim_sources else None

        rnn_input_size = _find_int(text, ("input_size", "inputSize"))
        rnn_hidden_size = _find_int(text, ("hidden_size", "hiddenSize"))
        logged_proj_size = _find_int(text, ("proj_size", "projSize"))
        seq_lengths = _find_sequence_lengths(text)
        explicit_layout = _find_layout(text)
        explicit_wgrad_mode = _find_wgrad_mode(text)

        result: Dict[str, Any] = {
            "source": source_name,
            "source_line": source_line,
            "function": function_name,
            "mode": SUPPORTED_RNN_FUNCTIONS[function_name],
            "seq_length": _find_int(
                text,
                ("seqLength", "seq_length", "maxSeqLength", "max_seq_length"),
            ),
            "algo": _clean_enum(_find_value(text, ("algo", "algorithm"))),
            "cell_mode": _clean_enum(_find_value(text, ("cell_mode", "cellMode"))),
            "bias_mode": _clean_enum(_find_value(text, ("bias_mode", "biasMode"))),
            "dir_mode": _clean_enum(_find_value(text, ("dir_mode", "dirMode"))),
            "input_mode": _clean_enum(_find_value(text, ("input_mode", "inputMode"))),
            "clip_mode": _clean_enum(_find_value(text, ("clip_mode", "clipMode"))),
            "clip_nan_opt": _clean_enum(
                _find_value(text, ("clip_nan_opt", "clipNanOpt"))
            ),
            "data_type_enum": _find_data_type_enum(text),
            "math_prec": _clean_enum(_find_value(text, ("math_prec", "mathPrec"))),
            "math_type": _clean_enum(_find_value(text, ("math_type", "mathType"))),
            "rnn_input_size": rnn_input_size,
            "rnn_hidden_size": rnn_hidden_size,
            "logged_proj_size": logged_proj_size,
            "proj_size": logged_proj_size,
            "num_layers": _find_int(text, ("num_layers", "numLayers")),
            "batch_size": batch_dims[0] if batch_dims else _find_int(
                text, ("batch_size", "batchSize")
            ),
            "batch_dims_source": batch_dims_source,
            # rnnDesc 是 input/hidden size 的权威来源；descriptor dims 只用于交叉检查。
            "input_size": rnn_input_size,
            "hidden_size": rnn_hidden_size,
            "output_size": y_dims[1] if y_dims and len(y_dims) >= 2 else None,
            "dropout": _find_float(text, ("drop_rate", "dropout", "dropout_rate")),
            "lclip": _find_float(text, ("lclip", "left_clip")),
            "rclip": _find_float(text, ("rclip", "right_clip")),
            "layout": _parse_layout(text),
            "wgrad_mode": _parse_wgrad_mode(text),
            "sequence_lengths": seq_lengths,
            "workspace_size": _find_int(
                text, ("workSpaceSizeInBytes", "workspace_size", "workSpaceSize")
            ),
            "reserve_space_size": _find_int(
                text, ("reserveSpaceSizeInBytes", "reserve_space_size")
            ),
            "weight_size": w_dims[0] if w_dims else None,
            "weight_grad_size": dw_dims[0] if dw_dims else None,
            "x_dims": x_dims,
            "dx_dims": dx_dims,
            "dy_dims": dy_dims,
            "hx_dims": hx_dims,
            "y_dims": y_dims,
        }

        # 执行 API 可能只打印 rnnDesc 指针，参数实际来自前面的 v6/v8 配置 API。
        # 调用本身的字段优先，只有缺失时才由最近一个 descriptor 补全。
        descriptor_fields = (
            "algo", "cell_mode", "bias_mode", "dir_mode", "input_mode",
            "clip_mode", "clip_nan_opt", "data_type_enum", "math_prec", "math_type",
            "rnn_input_size", "rnn_hidden_size", "logged_proj_size", "num_layers",
            "dropout", "lclip", "rclip", "layout", "wgrad_mode",
        )
        if descriptor is not None:
            for field in descriptor_fields:
                if result.get(field) is None and descriptor.get(field) is not None:
                    result[field] = descriptor[field]
            result["descriptor_api"] = descriptor["function"]
            result["descriptor_version"] = descriptor["version"]
            result["descriptor_source_line"] = descriptor["source_line"]
        else:
            result["descriptor_api"] = None
            result["descriptor_version"] = None
            result["descriptor_source_line"] = None

        # mcdnnRNNForward (v8) 常把 T/B 放在 mcdnnSetRNNDataDescriptor 中，
        # 而不是重新打印在执行 API 内。
        if data_descriptor is not None:
            for field in ("seq_length", "batch_size", "data_type_enum", "sequence_lengths"):
                if result.get(field) is None and data_descriptor.get(field) is not None:
                    result[field] = data_descriptor[field]
            if batch_dims is None and data_descriptor.get("batch_size") is not None:
                result["batch_dims_source"] = "rnn_data_descriptor"
            if explicit_layout is None and data_descriptor.get("layout") is not None:
                result["layout"] = data_descriptor["layout"]
            result["data_descriptor_api"] = data_descriptor["function"]
            result["data_descriptor_source_line"] = data_descriptor["source_line"]
        else:
            result["data_descriptor_api"] = None
            result["data_descriptor_source_line"] = None

        result["input_size"] = result["rnn_input_size"]
        result["hidden_size"] = result["rnn_hidden_size"]
        if result["proj_size"] is None:
            result["proj_size"] = result["logged_proj_size"]

        warnings: List[str] = []
        batch_values = {dims[0] for _, dims in batch_dim_sources}
        if len(batch_values) > 1:
            warnings.append(
                "batch descriptor mismatch: {}".format(
                    ", ".join(
                        "{}={}".format(name, dims[0])
                        for name, dims in batch_dim_sources
                    )
                )
            )
        if result["proj_size"] is None:
            result["proj_size"] = 0
        # 对测试 YAML 来说，P=0 表示未启用 projection。运行时 descriptor
        # 可能把未启用 projection 打印成 P=H；GRU 本身也不使用 LSTM projection。
        # 两种日志表示应归一化为同一个逻辑 case，避免生成 ...xHxH 的假 shape。
        if (
            result["cell_mode"] != "MCDNN_LSTM"
            or (
                result["rnn_hidden_size"] is not None
                and result["proj_size"] == result["rnn_hidden_size"]
            )
        ):
            if result["proj_size"] != 0:
                warnings.append(
                    "projection disabled: logged proj_size {} normalized to 0".format(
                        result["proj_size"]
                    )
                )
            result["proj_size"] = 0
        if result["dropout"] is None:
            result["dropout"] = 0.0
        if result["lclip"] is None:
            result["lclip"] = 0.0
        if result["rclip"] is None:
            result["rclip"] = 0.0

        result["data_type"] = precision_name(result["data_type_enum"])
        result["has_padding"] = bool(
            seq_lengths and any(value != seq_lengths[0] for value in seq_lengths)
        )

        input_dims = dx_dims if function_name == "mcdnnRNNBackwardData" else x_dims
        input_dims_name = "dx" if function_name == "mcdnnRNNBackwardData" else "x"
        if input_dims and len(input_dims) >= 2 and result["rnn_input_size"] is not None:
            if input_dims[1] != result["rnn_input_size"]:
                warnings.append(
                    f"{input_dims_name} descriptor input size {input_dims[1]} "
                    f"!= rnnDesc input_size {rnn_input_size}"
                )
        if hx_dims and len(hx_dims) >= 3 and result["rnn_hidden_size"] is not None:
            if hx_dims[2] != result["rnn_hidden_size"]:
                warnings.append(
                    f"hx descriptor hidden size {hx_dims[2]} != rnnDesc hidden_size {result['rnn_hidden_size']}"
                )

        errors = [
            f"缺少必需字段：{field}"
            for field in REQUIRED_FIELDS
            if result.get(field) is None
        ]
        for field in ("seq_length", "num_layers", "batch_size", "input_size", "hidden_size"):
            value = result.get(field)
            if value is not None and value <= 0:
                errors.append(f"字段 {field} 必须大于 0，实际为 {value}")
        if result["proj_size"] < 0:
            errors.append(f"字段 proj_size 不能小于 0，实际为 {result['proj_size']}")

        result["parse_warnings"] = warnings
        result["parse_errors"] = errors
        result["valid"] = not errors
        defaulted_fields: List[str] = []
        if logged_proj_size is None and (descriptor is None or descriptor.get("logged_proj_size") is None):
            defaulted_fields.append("proj_size")
        if _find_float(text, ("drop_rate", "dropout", "dropout_rate")) is None:
            defaulted_fields.append("dropout")
        if _find_float(text, ("lclip", "left_clip")) is None:
            defaulted_fields.append("lclip")
        if _find_float(text, ("rclip", "right_clip")) is None:
            defaulted_fields.append("rclip")
        if result["clip_mode"] is None:
            defaulted_fields.append("clip_mode")
        if result["clip_nan_opt"] is None:
            defaulted_fields.append("clip_nan_opt")
        if explicit_layout is None:
            defaulted_fields.append("layout")
        if explicit_wgrad_mode is None:
            defaulted_fields.append("wgrad_mode")
        if seq_lengths is None:
            defaulted_fields.append("sequence_lengths")
        result["defaulted_fields"] = defaulted_fields
        return result


def _strip_prefix(value: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def algo_name(value: str) -> str:
    stripped = _strip_prefix(value, ("MCDNN_RNN_ALGO_",))
    mapping = {
        "STANDARD": "stand",
        "PERSIST_STATIC": "persist_static",
        "PERSIST_STATIC_SMALL_H": "persist_static_small_h",
        "PERSIST_DYNAMIC": "persist_dynamic",
    }
    return mapping.get(stripped, _token(stripped))


def cell_name(value: str) -> str:
    return _token(_strip_prefix(value, ("MCDNN_RNN_", "MCDNN_")))


def bias_name(value: str) -> str:
    stripped = _strip_prefix(value, ("MCDNN_RNN_", "MCDNN_"))
    return _token(stripped).replace("_", "")


def direction_name(value: str) -> str:
    stripped = _strip_prefix(value, ("MCDNN_RNN_", "MCDNN_"))
    mapping = {
        "UNIDIRECTIONAL": "unidir",
        "BIDIRECTIONAL": "bidir",
    }
    return mapping.get(stripped, _token(stripped))


def input_mode_name(value: str) -> str:
    stripped = _strip_prefix(value, ("MCDNN_RNN_", "MCDNN_"))
    mapping = {
        "LINEAR_INPUT": "lininput",
        "SKIP_INPUT": "skipinput",
    }
    return mapping.get(stripped, _token(stripped).replace("_", ""))


def clip_name(value: Optional[str], nan_opt: Optional[str]) -> str:
    if value in (None, "MCDNN_RNN_CLIP_NONE", "MCDNN_CLIP_NONE"):
        return "clipnon"
    stripped = _strip_prefix(value, ("MCDNN_RNN_", "MCDNN_"))
    result = _token(stripped).replace("_", "")
    if nan_opt and "NOT_PROPAGATE_NAN" not in nan_opt:
        result += "prop"
    return result


def layout_name(value: str) -> str:
    mapping = {
        "MCDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED": "sequnpack",
        "MCDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED": "seqpack",
        "MCDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED": "batchunpack",
    }
    return mapping.get(value, _token(value).replace("_", ""))


def wgrad_name(value: str) -> str:
    upper = value.upper()
    if upper.endswith("_ADD") or upper == "ADD":
        return "wgradadd"
    if upper.endswith("_SET") or upper == "SET":
        return "wgradset"
    return "wgrad" + _token(value).replace("_", "")


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}" if value else "0.0"
    return str(value)


def shape_anchor(item: Dict[str, Any]) -> str:
    return (
        f"shape_rnn_{item['num_layers']}l_"
        f"{item['seq_length']}x{item['batch_size']}x{item['input_size']}x"
        f"{item['hidden_size']}x{item['proj_size']}"
    )


def algorithm_anchor(item: Dict[str, Any]) -> str:
    parts = (
        "rnn",
        algo_name(item["algo"]),
        item["data_type"],
        cell_name(item["cell_mode"]),
        bias_name(item["bias_mode"]),
        direction_name(item["dir_mode"]),
        input_mode_name(item["input_mode"]),
        clip_name(item.get("clip_mode"), item.get("clip_nan_opt")),
        layout_name(item["layout"]),
        wgrad_name(item["wgrad_mode"]),
    )
    return "_".join(parts)


class YAMLGenerator:
    """生成与 RNN 测试仓库约定一致的两个 YAML 片段。"""

    @staticmethod
    def _valid_functions(parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [item for item in parsed_data.get("functions", []) if item.get("valid")]

    @staticmethod
    def generate_shape_definitions(parsed_data: Dict[str, Any]) -> str:
        lines: List[str] = []
        seen: Dict[str, Tuple[Any, ...]] = {}

        for item in YAMLGenerator._valid_functions(parsed_data):
            anchor = shape_anchor(item)
            signature = (
                item["num_layers"],
                item["seq_length"],
                item["batch_size"],
                item["input_size"],
                item["hidden_size"],
                item["proj_size"],
                item["dropout"],
                item["lclip"],
                item["rclip"],
            )
            if anchor in seen:
                if seen[anchor] != signature:
                    raise ValueError(f"shape anchor 冲突：{anchor}")
                continue
            seen[anchor] = signature

            lines.append(f"- &{anchor}")
            lines.append(
                "  {<<: *default_rnn_shape, "
                f"rnn_num_layers: {item['num_layers']}, "
                f"rnn_max_seq_length: {item['seq_length']}, "
                f"rnn_batch_size: {item['batch_size']}, "
                f"rnn_input_size: {item['input_size']}, "
                f"rnn_hidden_size: {item['hidden_size']}, "
                f"rnn_proj_size: {item['proj_size']}, "
                f"rnn_dropout: {_number(item['dropout'])}, "
                f"rnn_lclip: {_number(item['lclip'])}, "
                f"rnn_rclip: {_number(item['rclip'])}}}"
            )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_rnn_config(parsed_data: Dict[str, Any]) -> str:
        lines: List[str] = []
        seen = set()

        for item in YAMLGenerator._valid_functions(parsed_data):
            algo = algorithm_anchor(item)
            shape = shape_anchor(item)
            shape_suffix = shape[len("shape_rnn_") :] if shape.startswith("shape_rnn_") else shape
            full_name = f"{algo}_{shape_suffix}"
            if full_name in seen:
                continue
            seen.add(full_name)

            tolerance = (
                "eps_large_2" if item["data_type"] in ("fp16", "bf16") else "eps_normal"
            )
            seq_length_anchor = (
                "seq_length_padding" if item["has_padding"] else "seq_length_no_padding"
            )
            lines.append(f"- name: {full_name}")
            lines.append(f"  algo: *{algo}")
            lines.append(f"  shape: *{shape}")
            lines.append(f"  seq_length: *{seq_length_anchor}")
            lines.append("  init_data: *data_gen_random_range_0p1")
            lines.append(f"  error_tolerance: *{tolerance}")
            lines.append("  frequency: {mcsim: *local, cmodel : *local}")
            lines.append("")

        return "\n".join(lines)


class OutputManager:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, parsed_data: Dict[str, Any], show_result: bool = False) -> None:
        json_file = self.output_dir / "parsed_data.json"
        shape_file = self.output_dir / "shape_definitions.yaml"
        config_file = self.output_dir / "rnn_config.yaml"

        json_file.write_text(
            json.dumps(parsed_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shape_file.write_text(
            YAMLGenerator.generate_shape_definitions(parsed_data), encoding="utf-8"
        )
        config_file.write_text(
            YAMLGenerator.generate_rnn_config(parsed_data), encoding="utf-8"
        )

        print(f"✓ 保存解析数据: {json_file}")
        print(f"✓ 保存Shape定义: {shape_file}")
        print(f"✓ 保存YAML配置: {config_file}")
        if show_result:
            self.display(parsed_data)

    @staticmethod
    def display(parsed_data: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print(
            f"来源: {parsed_data['total_sources']}，"
            f"RNN调用: {parsed_data['total_functions']}，"
            f"有效: {parsed_data['valid_functions']}，"
            f"失败: {len(parsed_data['parse_errors'])}"
        )
        for item in parsed_data["functions"]:
            status = "OK" if item["valid"] else "ERROR"
            print(
                f"[{status}] {item['source']}:{item['source_line']} "
                f"{item['function']} "
                f"T={item.get('seq_length')} B={item.get('batch_size')} "
                f"I={item.get('input_size')} H={item.get('hidden_size')}"
            )
            for error in item["parse_errors"]:
                print(f"  - {error}")


def generate_sample_log(filename: str) -> None:
    sample = """23 Jul 18:33 INFO unrelated application output
[MCDNN][INFO]
[MCDNN][INFO] function mcdnnSetRNNDescriptor_v6() called:
[MCDNN][INFO] rnnDesc:0x01
[MCDNN][INFO] algo:MCDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H,cell_mode:MCDNN_GRU,bias_mode:MCDNN_RNN_DOUBLE_BIAS,dir_mode:MCDNN_UNIDIRECTIONAL,input_mode:MCDNN_LINEAR_INPUT,data_type:MCDNN_DATA_HALF,math_prec:MCDNN_DATA_FLOAT,math_type:MCDNN_DEFAULT_MATH,dropout_desc:{drop_rate:0,states:0,seed:0},input_size:64,hidden_size:128,proj_size:0,num_layers:1,clip_mode:MCDNN_RNN_CLIP_NONE,clip_nan_opt:MCDNN_NOT_PROPAGATE_NAN,lclip:0,rclip:0
[MCDNN][INFO] function mcdnnRNNForwardInference() called:
[MCDNN][INFO] handle:0
[MCDNN][INFO] rnnDesc:0x01
[MCDNN][INFO] seqLength:18
[MCDNN][INFO] x_desc:data type:MCDNN_DATA_HALF,fmt:MCDNN_TENSOR_NCHW,nbdims:5dim:[8192,64,1,1,1]strides:[64,1,1,1,1]
[MCDNN][INFO] hxDesc:data type:MCDNN_DATA_HALF,fmt:MCDNN_TENSOR_NCHW,nbdims:5dim:[1,8192,128,1,1]strides:[1048576,128,1,1,1]
[MCDNN][INFO] wDesc:data type:MCDNN_DATA_HALF,fmt:MCDNN_TENSOR_NCHW,nbdims:3dim:[74496,1,1]
[MCDNN][INFO] y_desc:data type:MCDNN_DATA_HALF,fmt:MCDNN_TENSOR_NCHW,nbdims:5dim:[8192,128,1,1,1]strides:[128,1,1,1,1]
[MCDNN][INFO] workSpaceSizeInBytes:56623104
"""
    Path(filename).write_text(sample, encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "解析 mcDNN RNN API 日志，生成可追溯 JSON 与 YAML testcase 片段。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""输入说明：
  -i 可重复指定。每个输入可为普通日志文件、目录、zip、tar、tar.gz 或 tgz。
  目录会递归遍历；压缩包成员在内存中读取，不会解压到文件系统。

输出目录（由 -o 指定）包含：
  parsed_data.json          每次调用的来源、字段、告警和解析错误
  shape_definitions.yaml    shape 锚点定义片段
  rnn_config.yaml           RNN testcase 片段

解析规则：
  支持 v8 通用 RNNForward，以及 ForwardInference、ForwardTraining、
  BackwardData、BackwardFilter、BackwardWeights。GRU/非 LSTM 日志中的 proj_size=hidden_size 会按无
  projection（P=0）归一化；descriptor 的 3D/5D 打印差异不会单独导致失败。

示例：
  python3 parse_mcdnn_log.py -i ./gru_logs -o ./parsed_rnn -s
  python3 parse_mcdnn_log.py -i app.log -i archived_logs.tar.gz -o ./out
  python3 parse_mcdnn_log.py --sample

退出码：
  0  成功，所有识别到的 RNN 调用均包含必需字段
  1  输入读取失败，或存在缺少必需字段的 RNN 调用
  2  未找到支持的 RNN API 调用
  3  输出文件写入失败或生成 YAML 时发现冲突
""",
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        metavar="PATH",
        help="日志文件、目录或压缩包；可重复指定（默认：./log.txt）",
    )
    parser.add_argument(
        "-o", "--output", default="./output", metavar="DIR",
        help="输出目录（默认：./output）",
    )
    parser.add_argument(
        "-s", "--show", action="store_true",
        help="同时打印每次识别到的 RNN 调用摘要及其解析错误",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="在当前目录生成示例日志 log.txt 后退出",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.sample:
        generate_sample_log("log.txt")
        print("✓ 生成示例日志文件: log.txt")
        return 0

    input_paths = args.input or ["log.txt"]
    try:
        # 直接迭代输入来源，避免目录中多份大日志同时常驻内存。
        parsed_data = MCDNNRNNParser().parse_sources(iter_log_sources(input_paths))
    except (FileNotFoundError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"✗ 读取输入失败：{error}", file=sys.stderr)
        return 1

    print(f"✓ 读取 {parsed_data['total_sources']} 个文本来源")
    if parsed_data["total_functions"] == 0:
        related = parsed_data["seen_rnn_related_apis"]
        if related:
            print(
                "✗ 仅发现 RNN 初始化/配置 API，未发现可生成 testcase 的执行 API。\n"
                f"  已发现: {', '.join(related)}\n"
                "  需要继续截取包含 mcdnnRNNForward、mcdnnRNNForwardInference、"
                "mcdnnRNNForwardTraining、mcdnnRNNBackwardData 或 "
                "mcdnnRNNBackwardWeights 的日志。",
                file=sys.stderr,
            )
        else:
            print("✗ 未找到支持的 MCDNN RNN 函数调用", file=sys.stderr)
        return 2

    try:
        OutputManager(args.output).save(parsed_data, args.show)
    except (OSError, ValueError) as error:
        print(f"✗ 生成输出失败：{error}", file=sys.stderr)
        return 3

    if parsed_data["parse_errors"]:
        print(
            f"✗ {len(parsed_data['parse_errors'])} 个 RNN 调用缺少必需字段；"
            "已写入 parsed_data.json，但未生成对应 testcase。",
            file=sys.stderr,
        )
        return 4

    print(
        f"✓ 完成：{parsed_data['valid_functions']} 个有效 RNN 调用，"
        f"输出目录 {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
