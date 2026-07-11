import pytest

from app.core.runtime import require_supported_python


def test_python_312_is_supported():
    require_supported_python((3, 12, 13))


def test_python_314_is_rejected_with_actionable_message():
    with pytest.raises(RuntimeError, match=r"requires Python 3\.12\.x.*\.venv312"):
        require_supported_python((3, 14, 5))
