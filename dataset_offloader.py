# Dataset Loading Code
import pandas as pd
import json

df = pd.read_parquet("./math_problems.parquet")

problems = df["problem"].astype(str)
solutions = df["solution"].astype(str)

def extract_steps(solution):
    steps = [s.strip() for s in solution.split("\n") if s.strip()]
    return [
        {"step": i+1, "goal": step}
        for i, step in enumerate(steps)
    ]

dataset = []

for i in range(len(df)):
    problem = problems.iloc[i]
    solution = solutions.iloc[i]
    steps = extract_steps(solution)

    dataset.append({
        "problem": problem,
        "solution": steps
    })

with open("processed_dataset.json", "w") as f:
    json.dump(dataset, f, indent = 2)