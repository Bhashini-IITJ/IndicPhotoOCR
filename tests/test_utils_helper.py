"""
Tests for IndicPhotoOCR.utils.helper

detect_para() is a pure Python/NumPy function – no model weights needed.
All tests here run without any external downloads.
"""
import pytest
from IndicPhotoOCR.utils.helper import detect_para


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


def _make_dict(**entries):
    """Build a recognized_texts dict from keyword=((txt, bbox)) pairs."""
    result = {}
    for key, (txt, bbox) in entries.items():
        result[key] = {"txt": txt, "bbox": bbox}
    return result


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------

class TestDetectParaReturnType:
    def test_returns_list(self, sample_bbox_dict):
        result = detect_para(sample_bbox_dict)
        assert isinstance(result, list), "detect_para must return a list"

    def test_inner_elements_are_lists(self, sample_bbox_dict):
        result = detect_para(sample_bbox_dict)
        for line in result:
            assert isinstance(line, list), "Each line grouping must be a list"

    def test_inner_elements_are_strings(self, sample_bbox_dict):
        result = detect_para(sample_bbox_dict)
        for line in result:
            for word in line:
                assert isinstance(word, str), f"Expected str, got {type(word)}"

    def test_empty_input_returns_empty_list(self):
        result = detect_para({})
        assert result == []

    def test_single_word(self, single_word_bbox_dict):
        result = detect_para(single_word_bbox_dict)
        assert len(result) == 1
        assert result[0] == ["only"]


# ---------------------------------------------------------------------------
# Word preservation
# ---------------------------------------------------------------------------

class TestDetectParaWordPreservation:
    def test_all_words_are_preserved(self, sample_bbox_dict):
        result = detect_para(sample_bbox_dict)
        flat = [word for line in result for word in line]
        expected = {v["txt"] for v in sample_bbox_dict.values()}
        assert set(flat) == expected

    def test_no_words_are_duplicated(self, sample_bbox_dict):
        result = detect_para(sample_bbox_dict)
        flat = [word for line in result for word in line]
        assert len(flat) == len(sample_bbox_dict)

    def test_unicode_words_preserved(self):
        d = _make_dict(
            w0=("राजीव",  _bbox(0,  0,  60, 20)),
            w1=("चौक",    _bbox(70, 2,  130, 22)),
        )
        result = detect_para(d)
        flat = [w for line in result for w in line]
        assert "राजीव" in flat
        assert "चौक" in flat


# ---------------------------------------------------------------------------
# Line grouping / horizontal sorting
# ---------------------------------------------------------------------------

class TestDetectParaGrouping:
    def test_two_words_same_line_are_grouped(self):
        """Words with ≥40 % vertical overlap should end up in the same line."""
        d = _make_dict(
            left=("left",  _bbox(0,  10, 50, 30)),
            right=("right", _bbox(60, 10, 120, 30)),
        )
        result = detect_para(d)
        assert len(result) == 1, "Two overlapping words must form one line"
        assert result[0] == ["left", "right"]  # left comes first (smaller x1)

    def test_two_words_different_lines_are_separated(self):
        """Words with no vertical overlap must be on different lines."""
        d = _make_dict(
            top=("top",    _bbox(0, 0,  80, 20)),
            bottom=("bottom", _bbox(5, 40, 85, 60)),
        )
        result = detect_para(d)
        assert len(result) == 2, "Non-overlapping words must form separate lines"

    def test_words_within_line_sorted_left_to_right(self):
        """Within a line, words should be sorted by ascending x1."""
        d = _make_dict(
            c=("C", _bbox(200, 10, 250, 30)),
            a=("A", _bbox(0,   10, 50,  30)),
            b=("B", _bbox(100, 10, 150, 30)),
        )
        result = detect_para(d)
        assert len(result) == 1
        assert result[0] == ["A", "B", "C"]

    def test_three_line_paragraph(self):
        """Three clearly separated horizontal bands → three lines."""
        d = _make_dict(
            l1w1=("Line1Word1", _bbox(0,   0,  80,  20)),
            l1w2=("Line1Word2", _bbox(90,  2,  170, 22)),
            l2w1=("Line2Word1", _bbox(5,   40, 85,  60)),
            l3w1=("Line3Word1", _bbox(10,  80, 90,  100)),
        )
        result = detect_para(d)
        assert len(result) == 3

    def test_overlap_threshold_boundary(self):
        """Overlap exactly at 0.4 boundary: at-or-above merges, below separates."""
        # Box heights = 20 px each; overlap = 8 px → ratio = 8/20 = 0.4 → merges
        at_threshold = _make_dict(
            a=("A", _bbox(0,  0,  40, 20)),
            b=("B", _bbox(50, 12, 90, 32)),   # overlap = 8 px
        )
        result_at = detect_para(at_threshold)
        # overlap/height == 0.4 → condition is `> 0.4`, so these should be SEPARATE
        assert len(result_at) == 2, "Overlap exactly at 0.4 should NOT merge (strict >)"

        # overlap = 9 px → 9/20 = 0.45 → merges
        above_threshold = _make_dict(
            a=("A", _bbox(0,  0,  40, 20)),
            b=("B", _bbox(50, 11, 90, 31)),   # overlap = 9 px
        )
        result_above = detect_para(above_threshold)
        assert len(result_above) == 1, "Overlap > 0.4 should merge into one line"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestDetectParaEdgeCases:
    def test_zero_height_bbox_does_not_crash(self):
        """A degenerate zero-height box (y1 == y2) must not raise ZeroDivisionError."""
        d = _make_dict(
            degenerate=("oops", _bbox(0, 10, 50, 10)),  # h = 0
            normal=("fine",     _bbox(0, 30, 50, 50)),
        )
        # Should not raise
        result = detect_para(d)
        assert isinstance(result, list)

    def test_large_number_of_words(self):
        """50 words on a single horizontal line should all end in one group."""
        d = {}
        for i in range(50):
            d[f"img_{i}"] = {"txt": f"word{i}", "bbox": [i * 20, 5, i * 20 + 18, 25]}
        result = detect_para(d)
        assert len(result) == 1
        flat = result[0]
        assert len(flat) == 50
        # Words should be left-to-right
        indices = [int(w.replace("word", "")) for w in flat]
        assert indices == sorted(indices)
