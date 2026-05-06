import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("./math_problems.parquet")

print("Total problems:", len(df))

df["problem_length"] = df["problem"].apply(len)
df["solution_length"] = df["solution"].apply(len)

print("Problem length stats:")
print(df["problem_length"].describe())

print("Solution length stats:")
print(df["solution_length"].describe())

plt.hist(df["solution_length"], bins=30)
plt.title("Distribution of Solution Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()