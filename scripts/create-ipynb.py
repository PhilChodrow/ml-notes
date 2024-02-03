import os
import nbformat as nb
import re

os.makedirs("docs/live-notebooks", exist_ok=True)
for f in os.listdir("source"):
    if f[-4:] == ".qmd":
        out_path = f"docs/live-notebooks/{f.replace('qmd', 'ipynb')}"
        os.system(f"quarto convert source/{f} -o {out_path}")
        
        notebook = nb.read(out_path, as_version = 4)
        for cell in notebook["cells"]:
            cell["source"] = re.sub(r"#---[\S\s]*?#---", "", cell["source"])
        
        nb.write(notebook, out_path)
        
        
        
    

    

