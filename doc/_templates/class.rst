{{ objname | escape | underline}}

Import in python using `from {{ ".".join(fullname.split(".")[:-1]) }} import {{ objname }}`

.. rubric:: Documentation

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: methods
      :template: method.rst

   {% for item in methods %}
        {% if item not in ["__init__"] %}
      ~{{ name }}.{{ item }}
        {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}