from evaluate import incorporated_tool_result, is_exact_answer


def test_exact_answer_allows_whitespace_and_thousands_separators() -> None:
    assert is_exact_answer(" 107\n", "107")
    assert is_exact_answer(
        "INCORRECT: 9,999,399,973",
        "INCORRECT: 9999399973",
    )
    assert not is_exact_answer("The answer is 107", "107")


def test_incorporated_tool_result() -> None:
    assert incorporated_tool_result("107", "107", [{"result": 107}])
    assert incorporated_tool_result(
        "INCORRECT: 9999399973",
        "INCORRECT: 9999399973",
        [{"result": 9999399973}],
    )
    assert incorporated_tool_result(
        "INCORRECT: 9,999,399,973",
        "INCORRECT: 9999399973",
        [{"result": 9999399973}],
    )
    assert not incorporated_tool_result("107", "107", [])
