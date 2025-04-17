from openai import AzureOpenAI
from dotenv import load_dotenv
from gt import correct_category
from collections import defaultdict

import os
import re

load_dotenv()

# Key for Azure OpenAI API
AZURE_OPENAI_API_KEY = os.getenv('CLASS_AZURE_KEY')
MAX_TOKENS = 7000
LOG_PROBS = False

# The first model - gpt 3.5
AZURE_OPENAI_GPT35_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT')
GPT35_API_VERSION = '2023-12-01-preview'
MODEL_35 = 'gpt-35-16k'

#The second model - gpt 4 mini
AZURE_OPENAI_GPT40_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT_4o')
GPT40_API_VERSION = '2024-08-01-preview'
MODEL_40 = 'gpt-4o-mini'

temperatures = [0, .5, .9]
models = [35, 40]


gpt35_engine = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=GPT35_API_VERSION,
    azure_endpoint=AZURE_OPENAI_GPT35_ENDPOINT
)
gpt40_engine = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=GPT40_API_VERSION,
    azure_endpoint=AZURE_OPENAI_GPT40_ENDPOINT
)
def parse_split_response(response: str) -> str:
    r = response.lower()
    if "split" in r or "two" in r or "multiple issues" in r:
        return "split"
    elif "one" in r or "single issue" in r or "only one" in r or "no-split" in r:
        return "no-split"
    return "unknown"

def split():
    # Read the content of the tickets file
    with open("tkts_1.txt", "r") as file:
        file_content = file.read()
    # Use regex to extract each ticket
    tickets = re.findall(r"\d+:\s(.*?)(?=\n\d+:|\Z)", file_content, re.DOTALL)
    with open("split.txt", "w") as file:
        for i, ticket in enumerate(tickets, 1):
            file.write(str(i) + ":\n")
            system_prompt = {
                "role": "system",
                "content": "You are a professional support ticket splitter. your answers are limited to:'split', \n"
                           "'no-split'."
            }

            user_prompt = {
                "role": "user",
                "content": (
                    "I will give you a support ticket from my application. "
                    f"Please decide whether this ticket:{ticket} has one category, or more, from the follow list:\n"
                    "- interface\n"
                    "- lacking feature\n"
                    "- logic defect\n"
                    "- data\n"
                    "- security and access control\n"
                    "- configuration\n"
                    "- stability\n"
                    "- performance\n\n"
                    "Output only: split . if there is more then one category "
                    "else, output: no-split.\n\n"
                )
            }
            split_check = [[MODEL_35, 0], [MODEL_40, 0], [MODEL_40, .9]]
            msg = [system_prompt, user_prompt]
            for check in split_check:
                if check[0] == MODEL_35:
                    responses = gpt35_engine.chat.completions.create(messages=msg, model=MODEL_35, temperature=0,
                                                                     max_tokens=MAX_TOKENS, logprobs=LOG_PROBS, n=1)
                else:
                    responses = gpt40_engine.chat.completions.create(messages=msg, model=MODEL_40,
                                                                   temperature=check[1],
                                                                   max_tokens=MAX_TOKENS, logprobs=LOG_PROBS,
                                                                   n=1)
                # here i should write my decision according to the LLM's output.
                file.write(check[0] + ", " + str(check[1]) + ": " + parse_split_response(responses.choices[0].message.content) + "\n")
                file.write("LLM response: " + responses.choices[0].message.content + "\n")





def parse_category_response(response: str) -> str:
    CATEGORIES = [
        "interface", "lacking feature", "logic defect", "data",
        "security and access control", "configuration", "stability", "performance"
    ]
    found = [cat for cat in CATEGORIES if cat in response.lower()]
    if len(found) == 1:
        return found[0]
    elif len(found) == 2:
        return f"{found[0]} and {found[1]}"
    return "unknown"


def create_categories():
    # Read the content of the tickets file
    with open("tkts_2.txt", "r") as file:
        file_content = file.read()
    print("tkts_2 opened")
    # Extract numbered tickets
    tickets = re.findall(r"\d+:\s(.*?)(?=\n\d+:|\Z)", file_content, re.DOTALL)

    with open("categories.txt", "w") as file:
        for i, ticket in enumerate(tickets, 1):
            file.write(f"{i}:\n")

            system_prompt = {
                "role": "system",
                "content": "You are a professional support ticket categorizer."
            }

            user_prompt = {
                "role": "user",
                "content": (
                    "I will give you a support ticket from my application. "
                    "Please categorize it into one or two of the following categories:\n"
                    "- interface\n"
                    "- lacking feature\n"
                    "- logic defect\n"
                    "- data\n"
                    "- security and access control\n"
                    "- configuration\n"
                    "- stability\n"
                    "- performance\n\n"
                    "Output only the category name(s), all in lower case. "
                    "If two categories apply, connect them with ' and ' (e.g., 'data and logic defect').\n\n"
                    f"This is the ticket:\n{ticket}"
                )
            }

            msg = [system_prompt, user_prompt]

            for model in models:
                for temp in temperatures:
                    print(f"{model} with temp {temp} is running")
                    if model == 35:
                        response = gpt35_engine.chat.completions.create(
                            messages=msg, model=MODEL_35, temperature=temp,
                            max_tokens=MAX_TOKENS, logprobs=LOG_PROBS, n=1)
                        model_name = MODEL_35
                    else:
                        response = gpt40_engine.chat.completions.create(
                            messages=msg, model=MODEL_40, temperature=temp,
                            max_tokens=MAX_TOKENS, logprobs=LOG_PROBS, n=1)
                        model_name = MODEL_40

                    content = response.choices[0].message.content.strip()
                    parsed_category = parse_category_response(content)

                    file.write(f"{model_name}, {temp}: {parsed_category}\n")
                    file.write(f"LLM response: {content}\n\n")

def create_statistics():
    with open("categories.txt", "r") as f:
        lines = f.readlines()

    ticket_data = {}
    ticket_number = None
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.endswith(":") and line[:-1].isdigit():
            ticket_number = int(line[:-1])
            ticket_data[ticket_number] = []
        elif line.startswith("gpt"):
            match = re.match(r"(gpt-[\w\-]+), ([\d.]+): (.+)", line)
            if match:
                model, temp, parsed_category = match.groups()
                llm_response = lines[i+1].strip().replace("LLM response: ", "")
                ticket_data[ticket_number].append((model, float(temp), parsed_category.strip(), llm_response))

    # Now calculate stats
    with open("statistics.txt", "w") as out:
        correct_per_temp = defaultdict(int)
        total_per_temp = defaultdict(int)
        correct_per_model = defaultdict(int)
        total_per_model = defaultdict(int)
        majority_correct = 0
        total = 0

        for ticket_id, results in ticket_data.items():
            out.write(f"{ticket_id}:\n")
            correct_count = 0
            category_votes = defaultdict(int)

            for model, temp, category, _ in results:
                total += 1
                total_per_model[model] += 1
                total_per_temp[temp] += 1
                if correct_category(ticket_id, category):
                    correct_count += 1
                    correct_per_model[model] += 1
                    correct_per_temp[temp] += 1
                    category_votes[category] += 1

            out.write(f"number correct: {correct_count}\n")

            # majority voting
            majority_vote = max(category_votes.values(), default=0)
            if majority_vote >= 4:
                out.write("majority voting: correct\n\n")
                majority_correct += 1
            else:
                out.write("majority voting: incorrect\n\n")

        # Summary
        out.write("summary\n")
        for temp in [0, 0.5, 0.9]:
            pct = (correct_per_temp[temp] / total_per_temp[temp] * 100) if total_per_temp[temp] else 0
            out.write(f"percent correct temperature = {temp}: {pct:.2f}\n")

        for model in ["gpt-35-16k", "gpt-4o-mini"]:
            pct = (correct_per_model[model] / total_per_model[model] * 100) if total_per_model[model] else 0
            out.write(f"percent correct model = {model}: {pct:.2f}\n")

        total_correct = sum(correct_per_model.values())
        pct_total = total_correct / total * 100 if total > 0 else 0
        pct_majority = majority_correct / len(ticket_data) * 100 if ticket_data else 0

        out.write(f"percent correct total: {pct_total:.2f}\n")
        out.write(f"percent correct majority voting: {pct_majority:.2f}\n")




if __name__ == '__main__':
    split()
    print("finished splitting")
    create_categories()
    print("finished categorizing")
    create_statistics()
    print("done")





