# coding=utf-8

import re
from academicmarkdown import build
import biased_memory_toolbox as bmt
from npdoc_to_md import render_md_from_obj_docstring

api = render_md_from_obj_docstring(bmt.category, 'biased_memory_toolbox.category') \
    + '\n\n' + render_md_from_obj_docstring(bmt.fit_mixture_model, 'biased_memory_toolbox.fit_mixture_model') \
    + '\n\n' + render_md_from_obj_docstring(bmt.mixture_model_pdf, 'biased_memory_toolbox.mixture_model_pdf') \
    + '\n\n' + render_md_from_obj_docstring(bmt.prototype, 'biased_memory_toolbox.prototype') \
    + '\n\n' + render_md_from_obj_docstring(bmt.response_bias, 'biased_memory_toolbox.response_bias') \
    + '\n\n' + render_md_from_obj_docstring(bmt.test_chance_performance, 'biased_memory_toolbox.test_chance_performance')
build.postMarkdownFilters = []
with open('readme-template.md') as fd:
    text = fd.read()
for m in re.finditer('```python(?P<code>.*?)```', text, re.DOTALL):
    new_block = (
        u'\n%--\npython: |\n'
        + u'\n'.join([u' '+ s for s in m.group('code').strip().split(u'\n')])
        + u'\n--%\n'
    )
    text = text.replace(m.group(0), new_block)
md = build.MD(text)
md = md.replace('~~~ .python', '```python')
md = md.replace('\n~~~', '```')
md = md.replace('[API]', api)
with open('readme.md', 'w') as fd:
    fd.write(md)
