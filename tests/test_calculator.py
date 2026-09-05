import pytest

from calculator_agent import CalculatorError, calculate, run_calculator_tool


def test_calculate_basic_arithmetic() -> None:
    assert calculate("(17 * 23) / 4") == 97.75
    assert calculate("2 ** 8 + 3") == 259
    assert (
        calculate("(92837465019283 * 71628394517) - (498273645091 * 82736419) + 7391827465")
        == 6649757344982320222029647
    )


@pytest.mark.parametrize("expression", ["__import__('os')", "open('secret')", "1 // 2"])
def test_calculator_rejects_non_arithmetic(expression: str) -> None:
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_tool_call_returns_json() -> None:
    result = run_calculator_tool(
        {"function": {"name": "calculate", "arguments": '{"expression": "6 * 7"}'}}
    )
    assert result == '{"result": 42}'
