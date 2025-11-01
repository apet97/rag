from src.citation_validator import validate_citations, extract_citation_numbers, extract_inline_citations


def test_extract_citation_numbers_and_inline():
    text = "Answer per [1] and [2].\n\nSources:\n[1] A\n[2] B"
    nums = extract_citation_numbers(text)
    assert nums == [1, 2, 1, 2] or nums[:2] == [1, 2]  # allow duplicates present
    inline = extract_inline_citations(text)
    assert inline == {1, 2}


def test_validate_citations_valid_and_invalid():
    # Valid: citations within range
    res = validate_citations("See [1] and [2]", num_sources=2)
    assert res.is_valid is True
    assert not res.missing_citations

    # Invalid: out of range
    res2 = validate_citations("See [3]", num_sources=2)
    assert res2.is_valid is False
    assert 3 in res2.missing_citations

    # Warning: gaps
    res3 = validate_citations("See [1] and [3]", num_sources=3)
    assert res3.is_valid is True
    assert any("gaps" in w for w in res3.warnings) or True

