import json
import sympy
import os
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini" # cheap, quick model

# below code is for testing and reference

# response = client.chat.completions.create(
#     model = MODEL,
#     messages = [
#         {"role": "system", "content": "You are a helpful math tutor."},
#         {"role": "user", "content": "Solve: 2x + 3y = 12, -4x + 6y = -8"}
#     ]
# )

# print(response.choices[0].message.content)

# Dataset Loading Code
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

def generate_plan(problem):
    # need to format the code like this so the GPT model explicitly knowns how to format its answer so it does not outright solve the problem
    prompt = f"""
    Break this math problem down into step-by-step goals WITHOUT providing answers before the user guesses them and WITHOUT fully solving the problem.
    Be sure to solve the problem using ONE clear method. Do not offer alternatives, as this is a linear step-by-step path.

    Problem: {problem}

    Return a JSON of this EXACTLY this structure: 
    {{
      "steps": [
        {{"step": 1, "goal": "..."}},
        {{"step": 2, "goal": "..."}}
      ]
    }}
    """

    response = client.chat.completions.create(
        model = MODEL,
        response_format = {"type": "json_object"},
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# print(generate_plan("Solve: 2x + 3y = 12, -4x + 6y = -8"))

def ask_step(problem, step_goal):
    prompt = f"""
    You are a math tutor.

    Problem: {problem}
    Current Goal: {step_goal}

    Explain briefly what the student needs to do, then ask a question that makes them perform the step.

    NEVER provide the answer.
    """

    response = client.chat.completions.create(
        model = MODEL,
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def evaluate_answer(problem, step_goal, student_answer):
    prompt = f"""
    Problem: {problem}
    Step goal: {step_goal}
    Student answer: {student_answer}

    Decide if the student answer is correct/acceptable.

    Respond ONLY with valid JSON:
    {{
        "correct": true/false,
        "feedback": "brief praise if correct/short explanation or a hint if incorrect WITHOUT providing the answer"
        "expected_answer": "provide an answer that would be accepted as correct for this step"
    }}
    """

    response = client.chat.completions.create(
        model = MODEL,
        response_format = {"type": "json_object"},
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def tutor_session():
    problem = input("Enter your math problem here: ")
    plan = json.loads(generate_plan(problem))["steps"]

    print(f'plan: {plan}')
    
    current_step = 0

    while current_step < len(plan):
        question = ask_step(problem, plan[current_step]["goal"])
        print([current_step, plan[current_step]["goal"]])
        print(question)

        user_input = input("Your answer: ")

        result = json.loads(evaluate_answer(problem, plan[current_step]["goal"], user_input))

        print(f'result: {result}')
        if result['correct']:
            current_step += 1
            print("step increased")
        print(result["feedback"])
    print("Congratulations on solving the problem!")

tutor_session()