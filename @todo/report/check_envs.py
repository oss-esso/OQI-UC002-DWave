from pathlib import Path
from collections import Counter, defaultdict
p=Path('presentation_slides.tex')
s=p.read_text(encoding='utf-8')
import re
begs=re.findall(r'\\begin\{([^}]+)\}',s)
ends=re.findall(r'\\end\{([^}]+)\}',s)
cb=Counter(begs)
ce=Counter(ends)
all_envs=set(begs+ends)
print('Environment counts (begin vs end):')
for e in sorted(all_envs):
    print(f'{e}: begin={cb[e]}, end={ce[e]}')
# Find any ">
# Also check for unclosed braces inside tikzpicture specifically by extracting tikz blocks

