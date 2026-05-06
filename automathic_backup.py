import json
from sympy import sympify, simplify, Eq
import os
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini" # cheap, quick model

df = pd.read_parquet("./math_problems.parquet")

df = df.dropna(subset = ["problem", "solution"])
df["num_steps"] = df["solution"].apply(lambda x: x.count("\n") + 1)
df = df[df["num_steps"] > 1]

def generate_plan(problem):
    # need to format the code like this so the GPT model explicitly knowns how to format its answer so it does not outright solve the problem
    prompt = f"""
    Break this math problem into a STRICT linear sequence of steps.

    Rules:
    - Each step must specify EXACTLY ONE action.
    - Each step must define EXACTLY which variable or expression to operate on.
    - DO NOT allow choices (e.g., "choose", "either", "you can").
    - DO NOT mention alternative methods.
    - The student must follow ONE fixed path.

    Bad example:
    "Isolate one of the variables"

    Good example:
    "Isolate x from the first equation by adding 5y to both sides and dividing by 2"

    Problem: {problem}

    Return JSON:
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

def ask_step(problem, step_goal, previous_step_goal, previous_student_answer):
    prompt = f"""
    You are a math tutor.

    Problem: {problem}
    Current Goal: {step_goal}
    {f"Previously Completed Goal: {previous_step_goal}" if previous_step_goal is not None else ""}
    {f"Student Answer to Previous Goal: {previous_student_answer}" if previous_student_answer is not None else ""}

    Explain briefly what the student needs to do ONLY to complete this specific goal. Do not include any statements similar to "To complete this step/goal."
    {"""The student just completed the previously completed goal, so do not ask them to complete any steps relative to that goal.
    Use the student's answer to the previous goal as a reference for framing your explanation for this current goal.""" if previous_step_goal is not None else ""}

    NEVER provide the answer.
    """

    response = client.chat.completions.create(
        model = MODEL,
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Decide if the student answer is correct/acceptable. Evaluate this by deciding if the user successfully followed the instructions described in the step goal and submitted an answer that completes that step.
# Assume the user's answer will be short, but with the appropriate mathematical representation of what state the problem should be in after completing that step.

def evaluate_answer(problem, step_goal, student_answer):
    prompt = f"""
    Problem: {problem}
    Step goal: {step_goal}

    Briefly and precisely fill out the JSON below.

    Respond ONLY with valid JSON:
    {{
        "feedback": "assuming the student is incorrect, provide a little feedback and/or a hint that will help them towards the answer to the step",
        "expected_answer": "provide the correct answer to this step."
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

def normalize_expression(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("^", "**")  # allow student-friendly exponent syntax

    # If it's an equation like "x = 2", take the right-hand side
    if "=" in expr:
        expr = expr.split("=")[-1]

    return expr

def parse_equation(expr: str):
    expr = expr.strip().replace("^", "**")

    if "=" in expr:
        left, right = expr.split("=")
        return sympify(left), sympify(right)
    else:
        # treat as expression = 0
        e = sympify(expr)
        return e, 0

def is_equivalent(a: str, b: str) -> bool:
    try:
        a_l, a_r = parse_equation(a)
        b_l, b_r = parse_equation(b)

        # Move everything to one side
        expr1 = simplify(a_l - a_r)
        expr2 = simplify(b_l - b_r)

        # Check if their difference simplifies to 0
        return simplify(expr1 - expr2) == 0

    except Exception as e:
        print(f"Sympy error: {e}")
        return False

def tutor_session():
    problem = input("Enter your math problem here: ")
    plan = json.loads(generate_plan(problem))["steps"]
    previous_student_answer = None

    print(f'plan: {plan}')
    
    current_step = 0

    while current_step < len(plan):
        question = ask_step(problem, plan[current_step]["goal"], previous_step_goal = plan[current_step - 1]["goal"] if (current_step - 1) >= 0 else None, previous_student_answer = previous_student_answer)
        print([current_step, plan[current_step]["goal"]])
        print(question)

        user_input = input("Your answer: ")

        result = json.loads(evaluate_answer(problem, plan[current_step]["goal"], user_input))

        expected = result["expected_answer"]

        correct = is_equivalent(user_input, expected)

        print(f"Expected answer: {expected}")
        print(f"Student answer: {user_input}")
        print(f"Correct (SymPy): {correct}")

        if correct:
            current_step += 1
            print("step increased")
            previous_student_answer = user_input
            print("Nice work.")
        else:
            print(result["feedback"])
    print("Congratulations on solving the problem!")

tutor_session()