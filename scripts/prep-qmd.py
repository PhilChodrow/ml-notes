import os

if not os.path.isdir("docs/live-notebooks"):
    os.makedirs("docs/live-notebooks", exist_ok=True)
    
if not os.path.isdir("chapters"):
    os.makedirs("chapters", exist_ok=True)

for f in os.listdir("source"):
    if f[-4:] == ".qmd":
        with open(f"source/{f}", "r") as file: 
            cleaned = file.read().replace('#---\n', '')
            
            live_path = f"{f.replace('.qmd', '.ipynb')}"
            if live_path in os.listdir("docs/live-notebooks"):
                
                link_line = "*Download the live notebook corresponding to these notes [here](blank).* \n"
                
                cleaned_lines = cleaned.split("\n")
                ix = 0
                found = False
                while not found: 
                    line = cleaned_lines[ix]
                    if line: 
                        found = "#" in line 
                    ix += 1
                ix += 1    
                
                
                cleaned_lines.insert(ix, link_line)
                cleaned = "\n".join(cleaned_lines)
            
            with open(f"chapters/{f}", "w") as new: 
                new.write(cleaned)
            
        # os.system(f"quarto convert {f}")

