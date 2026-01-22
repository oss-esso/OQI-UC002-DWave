from pathlib import Path
p=Path('presentation_slides.tex')
s=p.read_text(encoding='utf-8')
for i,line in enumerate(s.splitlines(),1):
    if '\\begin{column}' in line or '\\end{column}' in line:
        print(f'{i:04}: {line}')
