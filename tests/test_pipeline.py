from pathlib import Path

import pandas as pd
import pytest

from flower_ugc_analysis.pipeline import (
    aggregate_categories,
    analyze,
    build_word_frequency,
    validate_schema,
)


class FakeSegmenter:
    name = "test-double"

    def open(self):
        return None

    def close(self):
        return None

    def segment(self, text):
        return text.split()

    def keywords(self, text, limit):
        return text.split()[:limit]


def sample_frame():
    return pd.DataFrame(
        {
            "原始文本": ["鲜花 包装 完整", "物流 服务 及时"],
            "平台": ["淘宝", "小红书"],
            "一级维度": ["包装设计", "物流配送"],
            "情感倾向": ["正向", "中性"],
            "内容类型": ["评论", "笔记"],
        }
    )


def test_validate_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="缺少必需字段"):
        validate_schema(pd.DataFrame({"原始文本": ["示例"]}))


def test_aggregate_categories_excludes_raw_text():
    result = aggregate_categories(sample_frame())
    assert set(result.columns) == {"统计字段", "类别", "数量"}
    assert "原始文本" not in set(result["统计字段"])


def test_build_word_frequency_filters_stopwords():
    result = build_word_frequency([["鲜花", "包装", "的"], ["鲜花", "物流"]])
    assert result.iloc[0].to_dict() == {"词语": "鲜花", "频次": 2}
    assert "的" not in set(result["词语"])


def test_analyze_writes_expected_outputs(tmp_path: Path):
    metadata = analyze(sample_frame(), tmp_path, FakeSegmenter(), keyword_limit=2)
    assert metadata["rows"] == 2
    assert {path.name for path in tmp_path.iterdir()} == {
        "segmented.csv",
        "word_frequency.csv",
        "category_summary.csv",
        "category_summary.svg",
        "analysis_metadata.json",
    }
    chart = (tmp_path / "category_summary.svg").read_text(encoding="utf-8")
    assert "鲜花 包装 完整" not in chart
    assert "分类汇总图" in chart
