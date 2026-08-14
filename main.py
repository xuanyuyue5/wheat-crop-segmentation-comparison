import os
from experiment_runner import run_all_model_experiments

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

all_results = run_all_model_experiments()

print("-" * 60)
print("Finished all model experiments.")
for model_title, sections in all_results.items():
    print("-" * 60)
    print(model_title)
    for section_title, dataframe in sections.items():
        print(section_title)
        print(dataframe.to_string(index=False))
