import pytest

from evalplat import __version__
from evalplat.cli import main


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 2
    assert "usage: evalplat" in capsys.readouterr().out
