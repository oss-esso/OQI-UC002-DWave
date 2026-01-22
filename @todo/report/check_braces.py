from pathlib import Path
p=Path('presentation_slides.tex')
s=p.read_text(encoding='utf-8')
bal=0
last_nonzero=None
for i,line in enumerate(s.splitlines(),1):
    for j,ch in enumerate(line,1):
        if ch=='{': bal+=1
        elif ch=='}': bal-=1
        if bal<0:
            print(f'NEGATIVE at line {i} col {j} bal={bal}')
            bal=0
    if bal!=0:
        last_nonzero=(i,bal)
if bal!=0:
    print('FINAL_BALANCE',bal,'last_nonzero_line',last_nonzero)
else:
    print('BALANCED')
# print last 80 lines for context
lines=s.splitlines()
for ln in range(max(1,len(lines)-80),len(lines)+1):
    print(f"{ln:04}: {lines[ln-1]}")
