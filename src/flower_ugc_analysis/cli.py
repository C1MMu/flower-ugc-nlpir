"""Command-line interface for the public research showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PyNLPIRSegmenter, analyze, load_table, validate_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="云南鲜切花 UGC 的 NLPIR 分词与聚合分析工具"
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV 或 Excel 输入文件")
    parser.add_argument("--output", required=True, type=Path, help="私有输出目录")
    parser.add_argument(
        "--keyword-limit", type=int, default=10, help="每条文本提取的关键词上限"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="只校验字段，不启动 NLPIR"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.keyword_limit < 1:
        raise SystemExit("--keyword-limit 必须大于 0")

    frame = load_table(args.input)
    validate_schema(frame)
    if args.validate_only:
        print(f"字段校验通过：{len(frame)} 行，{len(frame.columns)} 列")
        return

    metadata = analyze(
        frame=frame,
        output_dir=args.output,
        segmenter=PyNLPIRSegmenter(),
        keyword_limit=args.keyword_limit,
    )
    print(f"分析完成：{metadata['rows']} 行；输出目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
