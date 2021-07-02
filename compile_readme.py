# coding=utf-8

import re
from academicmarkdown import build

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
with open('readme.md', 'w') as fd:
    fd.write(md)
