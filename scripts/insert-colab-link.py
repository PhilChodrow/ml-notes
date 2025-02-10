import os 
import re

for f in os.listdir("docs/chapters"):
    if f.endswith(".html"):
        pattern = r'(<div class="quarto-title-meta">[\s\S]*</header>)'

        with open(f"docs/chapters/{f}", "r") as file: 
            
            notebook_path = f.replace(".html", ".ipynb")
            
            colab_link_line = f'<p><a href="http://colab.research.google.com/github/philchodrow/ml-notes/blob/main/docs/live-notebooks/{notebook_path}">Open the live notebook in Google Colab</a>.</p>'
            
            text = file.read()
            text = re.sub(pattern, r'\1\n\n' + colab_link_line, text)
            with open(f"docs/chapters/{f}", "w") as new: 
                new.write(text)

