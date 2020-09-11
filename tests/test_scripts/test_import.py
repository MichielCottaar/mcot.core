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


@pytest.fixture('script_name', get_names())
def test_import(script_name):
    load_all_mcot()
    scripts = directories.all_scripts()

    for module in script_dir:
        if module in (
            'gcoord.gui',
        ):
             continue
        print(module)
        filename, _ = script_dir.get(module.split('.'))

        script = importlib.import_module(filename)

        assert xor(
            hasattr(script, 'main'),
            hasattr(script, 'run_from_args') and hasattr(script, 'add_to_parser')
        ), "Found no or multiple ways to call the fuunction"
        if not hasattr(script, 'main'):
            parser = ArgumentParser()
            assert getattr(script, 'add_to_parser')(parser) is None
            assert isinstance(parser, ArgumentParser)

        with pytest.raises(SystemExit):
            script_dir([module, '-h'])
