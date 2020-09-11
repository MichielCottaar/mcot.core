#!/usr/bin/env python
# Script to publish the documentation on jalapeno

from fsl.utils.run import run
import os.path as op
from sphinx.cmd.build import main

main(('doc', 'doc/html'))
target = op.expanduser("~/www/doc/mcot/core")
if op.isdir(target):
    run(f"rsync -aP doc/html/ {target}/")

