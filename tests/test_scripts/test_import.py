from mcot.core.scripts import directories, load_all_mcot
import importlib
from argparse import ArgumentParser
from operator import xor
import pytest

load_all_mcot()


def get_names(scripts=None):
    if scripts is None:
        scripts = directories.all_scripts()
    if isinstance(scripts, dict):
        res = []
        for name in scripts.values():
            res.extend(get_names(name))
        return res
    return [scripts]


@pytest.mark.parametrize('module', get_names())
def test_import(module):

    if module in (
        'mcot.core._scripts.gcoord.gui',
    ):
         return
    print(module)
    script = importlib.import_module(module)

    assert xor(
        hasattr(script, 'main'),
        hasattr(script, 'run_from_args') and hasattr(script, 'add_to_parser')
    ), "Found no or multiple ways to call the fuunction"
    if not hasattr(script, 'main'):
        parser = ArgumentParser()
        assert getattr(script, 'add_to_parser')(parser) is None
        assert isinstance(parser, ArgumentParser)

    with pytest.raises(SystemExit):
        directories([module, '-h'])
