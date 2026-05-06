import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

df = pd.read_parquet("./math_problems.parquet")
df = df.dropna(subset=["problem", "solution"])

def retrieve_reference(problem):
    for _, row in df.iterrows():
        if problem.strip() == row["problem"].strip():
            return row["solution"]

    # in case better reference check doesnt work
    for _, row in df.iterrows():
        if problem.split()[0] in row["problem"]:
            return row["solution"]

    return None

def generate_plan(problem, reference_solution=None):
    prompt = f"""
    Break this math problem into a STRICT step-by-step solution plan.

    Rules:
    - Each step = exactly ONE operation
    - No branching or alternatives
    - Deterministic sequence only

    Problem:
    {problem}
    """

    if reference_solution:
        prompt += f"""
        Reference solution (for pattern guidance only, DO NOT copy directly):
        {reference_solution}
        """

        prompt += """
        Return JSON:
        {
          "steps": [
            {"step": 1, "goal": "..."},
            {"step": 2, "goal": "..."}
          ]
        }
        """

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.choices[0].message.content)["steps"]

def ask_step(problem, step_goal, prev_goal=None, prev_answer=None):
    prompt = f"""
    You are a math tutor.

    Problem: {problem}
    Current step goal: {step_goal}

    {f"Previous step: {prev_goal}" if prev_goal else ""}
    {f"Student answer: {prev_answer}" if prev_answer else ""}

    Explain ONLY what the student should do for this step.
    Do NOT solve the problem.
    Be concise.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def evaluate_step(problem, step_goal, student_answer):
    prompt = f"""
    You are a math tutor.

    Problem: {problem}
    Step goal: {step_goal}
    Student answer: {student_answer}

    Return JSON ONLY:
    {{
      "feedback": "short hint or encouragement"
    }}
    """

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.choices[0].message.content)

def tutor_session():
    problem = input("Enter your math problem:\n")
    reference_solution = retrieve_reference(problem) # using the dataset to 

    # if reference_solution:
    #     print("found reference solution")
    # else:
    #     print("no reference found")

    plan = generate_plan(problem, reference_solution)
    # for p in plan:
    #     print(p)

    current_step = 0

    while current_step < len(plan):
        step_goal = plan[current_step]["goal"]

        print(f"\n--- Step {current_step + 1} ---")
        print(step_goal)

        # hint = ask_step(problem, step_goal, prev_goal, prev_answer)
        # print("\nHint:")
        # print(hint)

        user_input = input("\nYour answer: ")

        feedback = evaluate_step(problem, step_goal, user_input)

        print(feedback["feedback"])

        current_step += 1

    print("\nProblem complete!")


tutor_session()