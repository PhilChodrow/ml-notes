import os
for f in os.listdir("source"):
    with open(f"source/{f}", "r") as file: 
        cleaned = file.read().replace('#-#\n', '')
        with open(f"chapters/{f}", "w") as new: 
            new.write(cleaned)
        
    # os.system(f"quarto convert {f}")


