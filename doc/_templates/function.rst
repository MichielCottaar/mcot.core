{{ objname | escape | underline}}

Import in python using `from {{ ".".join(fullname.split(".")[:-1]) }} import {{ objname }}`

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}