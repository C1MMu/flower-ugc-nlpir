"""Privacy-aware aggregation and NLPIR segmentation pipeline."""

from __future__ import annotations

import datetime as dt
import html
import json
from collections import Counter
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd


REQUIRED_COLUMNS = ("原始文本", "平台", "一级维度", "情感倾向", "内容类型")
CATEGORY_COLUMNS = ("平台", "一级维度", "情感倾向", "内容类型")
STOP_WORDS = frozenset(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 "
    "这个 那个 可以 但是 因为 所以 如果 已经 可能 应该 需要 觉得 比较 非常 特别 "
    "真的 其实 以及 或者 此外 同时 nan none".split()
)


class Segmenter(Protocol):
    """Minimal interface needed by the analysis pipeline."""

    name: str

    def open(self) -> None: ...

    def close(self) -> None: ...

    def segment(self, text: str) -> Sequence[str]: ...

    def keywords(self, text: str, limit: int) -> Sequence[str]: ...


class PyNLPIRSegmenter:
    """Adapter that keeps third-party PyNLPIR usage isolated."""

    name = "PyNLPIR 0.6.1 / NLPIR"

    def __init__(self) -> None:
        self._module = None

    def open(self) -> None:
        try:
            import pynlpir
        except ImportError as exc:
            raise RuntimeError("PyNLPIR 未安装，请先执行 python -m pip install -e .") from exc

        self._module = pynlpir
        try:
            pynlpir.open()
        except Exception as exc:
            raise RuntimeError(
                "NLPIR 初始化失败。请先执行 pynlpir update，并确认操作系统、架构与许可证兼容。"
            ) from exc

    def close(self) -> None:
        if self._module is not None:
            self._module.close()
            self._module = None

    def segment(self, text: str) -> Sequence[str]:
        if self._module is None:
            raise RuntimeError("分词器尚未初始化")
        return self._module.segment(text, pos_tagging=False)

    def keywords(self, text: str, limit: int) -> Sequence[str]:
        if self._module is None:
            raise RuntimeError("分词器尚未初始化")
        return self._module.get_key_words(text, max_words=limit, weighted=False) or []


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV or Excel without mutating the source file."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("仅支持 .csv、.xlsx 或 .xls 文件")


def validate_schema(frame: pd.DataFrame) -> None:
    """Reject inputs that cannot support the documented analysis."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少必需字段: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("输入文件没有数据行")
    if frame["原始文本"].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError("原始文本列没有可分析内容")


def aggregate_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sample-level counts without copying the original text."""

    blocks: list[pd.DataFrame] = []
    for column in CATEGORY_COLUMNS:
        counts = (
            frame[column]
            .fillna("未标注")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("类别")
            .reset_index(name="数量")
        )
        counts.insert(0, "统计字段", column)
        blocks.append(counts)
    return pd.concat(blocks, ignore_index=True)


def build_word_frequency(segmented_rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Count domain words after removing a compact, auditable stop-word set."""

    counter: Counter[str] = Counter()
    for tokens in segmented_rows:
        counter.update(
            token.strip()
            for token in tokens
            if len(token.strip()) >= 2 and token.strip().lower() not in STOP_WORDS
        )
    return pd.DataFrame(counter.most_common(), columns=["词语", "频次"])


def write_category_chart(summary: pd.DataFrame, destination: Path) -> None:
    """Export a compact SVG bar chart from aggregate counts only.

    The chart deliberately consumes ``category_summary.csv`` rather than raw
    UGC, so the public-facing figure never embeds the source texts.
    """

    width, height = 960, 560
    margin_left, margin_right, margin_top, margin_bottom = 210, 72, 86, 72
    chart_width = width - margin_left - margin_right
    candidates = (
        summary.sort_values(["数量", "类别"], ascending=[False, True])
        .head(12)
        .reset_index(drop=True)
    )
    max_count = max(int(candidates["数量"].max()), 1) if not candidates.empty else 1
    row_height = max(30, min(40, (height - margin_top - margin_bottom) // max(len(candidates), 1)))
    bar_height = max(14, row_height - 12)

    rows: list[str] = []
    for index, row in candidates.iterrows():
        count = int(row["数量"])
        label = f"{row['统计字段']} · {row['类别']}"
        y = margin_top + index * row_height
        bar_width = round(chart_width * count / max_count)
        rows.extend(
            [
                f'<text x="{margin_left - 16}" y="{y + 18}" text-anchor="end" class="label">{html.escape(label)}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" class="bar"/>',
                f'<text x="{margin_left + bar_width + 10}" y="{y + 18}" class="value">{count}</text>',
            ]
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">分类汇总图</title>
  <desc id="desc">由分类汇总数据生成的条形图，不含原始文本。</desc>
  <style>
    .background {{ fill: #ffffff; }}
    .title {{ fill: #111111; font: 700 26px Arial, Helvetica, sans-serif; }}
    .note {{ fill: #666666; font: 14px Arial, Helvetica, sans-serif; }}
    .label, .value {{ fill: #202020; font: 14px Arial, Helvetica, sans-serif; }}
    .bar {{ fill: #111111; }}
    .axis {{ stroke: #d0d0d0; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" class="background"/>
  <text x="48" y="46" class="title">分类汇总图</text>
  <text x="48" y="70" class="note">仅使用聚合计数生成，不包含任何原始文本</text>
  <line x1="{margin_left}" y1="{margin_top - 12}" x2="{margin_left}" y2="{height - margin_bottom + 8}" class="axis"/>
  {''.join(rows)}
</svg>'''
    destination.write_text(svg, encoding="utf-8")


def analyze(
    frame: pd.DataFrame,
    output_dir: Path,
    segmenter: Segmenter,
    keyword_limit: int = 10,
) -> dict[str, object]:
    """Run segmentation and export both private and aggregate outputs."""

    validate_schema(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    segmented_rows: list[list[str]] = []
    keyword_rows: list[list[str]] = []

    segmenter.open()
    try:
        for raw in frame["原始文本"].fillna("").astype(str):
            text = raw.strip()
            if not text:
                segmented_rows.append([])
                keyword_rows.append([])
                continue
            segmented_rows.append(list(segmenter.segment(text)))
            keyword_rows.append(list(segmenter.keywords(text, keyword_limit)))
    finally:
        segmenter.close()

    private_output = frame.copy()
    private_output["原始文本_分词"] = [" / ".join(tokens) for tokens in segmented_rows]
    private_output["原始文本_关键词"] = ["、".join(words) for words in keyword_rows]
    private_output.to_csv(output_dir / "segmented.csv", index=False, encoding="utf-8-sig")

    word_frequency = build_word_frequency(segmented_rows)
    word_frequency.to_csv(output_dir / "word_frequency.csv", index=False, encoding="utf-8-sig")

    category_summary = aggregate_categories(frame)
    category_summary.to_csv(output_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    write_category_chart(category_summary, output_dir / "category_summary.svg")

    metadata = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "required_columns": list(REQUIRED_COLUMNS),
        "segmenter": segmenter.name,
        "keyword_limit": keyword_limit,
        "method_boundary": (
            "NLPIR 仅用于分词与关键词提取；情感倾向为外部规则程序及人工复核后的输入字段。"
        ),
        "privacy_notice": "segmented.csv 含输入文本，只能保存在私有目录，不得提交到公开仓库。",
        "chart_notice": "category_summary.svg 仅由汇总计数生成，不包含原始文本。",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
