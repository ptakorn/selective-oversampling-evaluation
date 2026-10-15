"""Combine existing numerical table sources; no model fitting."""
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'reports';O.mkdir(exist_ok=True)
a=(R/'evidence_supplement/supplement_body.tex').read_text();b=(R/'external_control/external_tables.tex').read_text()
b=b.replace('\\section*{External Hybrid-Method Follow-up}','\\section*{S5. External Hybrid-Method Follow-up}').replace('This separate addendum uses E-numbered tables.','The following tables extend the supplementary evidence with Tables S16--S23.').replace('Tables E1--E4','Tables S16--S19').replace('E5--E8','S20--S23')
assert (a+b).count('\\begin{table}')==23
(O/'supplement_body.tex').write_text(a+'\n\\clearpage\n'+b)
(O/'supplement.tex').write_text('\\documentclass[10pt]{article}\n\\usepackage[a4paper,margin=18mm]{geometry}\n\\usepackage{amsmath,graphicx}\n\\renewcommand{\\thetable}{S\\arabic{table}}\n\\setlength{\\emergencystretch}{2em}\n\\begin{document}\n\\input{supplement_body.tex}\n\\end{document}\n')
print('23 table sources combined. Compile reports/supplement.tex from reports/ to rebuild its PDF.')
