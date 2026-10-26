# -*- coding: utf-8 -*-
"""rvcx —— RVC 换声工具链（训练编排 / 换声 / 客观评测）。

子模块按需导入：`from rvcx import paths` / `from rvcx import voice`。
这里不预先 import，避免 `python -m rvcx.paths` 触发重复导入告警。
"""
__version__ = "0.1.0"

__all__ = ["paths", "voice", "__version__"]
