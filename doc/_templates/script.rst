{{ ' '.join(fullname.split('.')[2:]) | underline}}

.. rubric:: Usage

Run using `mc_script {{ ' '.join(fullname.split('.')[2:]) }}` after installation
or using `~ndcn0236/bin/mc_script {{ ' '.join(fullname.split('.')[2:]) }}` on jalapeno.

Import in python using `from {{ ".".join(fullname.split(".")[:-1]) }} import {{ objname }}`

.. rubric:: Documentation

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree: functions
      :template: function.rst

   {% for item in functions %}
        {% if item not in ["main", "run_from_args", "add_to_parser"] %}
      {{ item }}
        {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. rubric:: CLI interface

   .. autosummary::
      :toctree: functions
      :template: function.rst

   {% for item in functions %}
        {% if item in ["main", "run_from_args", "add_to_parser"] %}
      {{ item }}
        {% endif %}
   {%- endfor %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree: classes
      :template: class.rst

   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
