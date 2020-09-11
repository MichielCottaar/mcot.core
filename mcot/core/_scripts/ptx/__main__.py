"""
Common interface to run all of the scripts
"""
from mcot.core.scripts import ScriptDirectory
import os.path as op


def main():
    """
    Runs one of the scripts from the command line
    """
    ScriptDirectory(op.split(__file__)[0])()


if __name__ == '__main__':
    main()