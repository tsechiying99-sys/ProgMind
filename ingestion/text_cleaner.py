"""PDF 文本归一化、页眉页脚清理和质量估计。"""

import re
import unicodedata
from collections import Counter
from typing import Iterable, List

from .models import ParsedPage


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_ENGLISH_HYPHEN_BREAK_RE = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
_MEANINGFUL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_GARBAGE_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？；：、,.!?;:'\"()（）\[\]【】<>《》%+\-*/_=]")


def clean_text(text: str) -> str:
    """执行保守清洗，不改写课程内容本身。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = _CONTROL_RE.sub("", text)
    text = _ENGLISH_HYPHEN_BREAK_RE.sub("", text)
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def text_quality(text: str) -> float:
    """根据有效字符、乱码比例和文本多样性估算 0~1 质量分。"""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0

    meaningful = len(_MEANINGFUL_RE.findall(compact))
    meaningful_ratio = meaningful / len(compact)
    garbage_ratio = len(_GARBAGE_RE.findall(compact)) / len(compact)
    unique_ratio = len(set(compact)) / len(compact)

    # 短文本的多样性天然偏低，因此只把极端重复视为明显降分项。
    diversity = min(1.0, unique_ratio / 0.18)
    length_factor = min(1.0, meaningful / 40.0)
    score = (
        0.55 * meaningful_ratio
        + 0.20 * (1.0 - min(1.0, garbage_ratio * 4.0))
        + 0.15 * diversity
        + 0.10 * length_factor
    )
    return round(max(0.0, min(1.0, score)), 4)


def remove_repeated_margins(pages: Iterable[ParsedPage]) -> List[ParsedPage]:
    """删除跨页高频出现的短页眉/页脚，至少三页时才启用。"""
    page_list = list(pages)
    if len(page_list) < 3:
        return page_list

    margin_lines = []
    page_lines = []
    for page in page_list:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        page_lines.append(lines)
        candidates = set(lines[:2] + lines[-2:])
        margin_lines.extend(line for line in candidates if len(line) <= 100)

    counts = Counter(margin_lines)
    threshold = max(3, int(len(page_list) * 0.6 + 0.999))
    repeated = {line for line, count in counts.items() if count >= threshold}
    if not repeated:
        return page_list

    for page, lines in zip(page_list, page_lines):
        page.text = clean_text("\n".join(line for line in lines if line not in repeated))
        page.quality_score = text_quality(page.text)
    return page_list
