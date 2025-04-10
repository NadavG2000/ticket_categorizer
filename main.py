from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Key for Azure OpenAI API
AZURE_OPENAI_API_KEY = os.getenv('CLASS_AZURE_KEY')
MAX_TOKENS = 7000
LOG_PROBS= False

# The first model - gpt 3.5
AZURE_OPENAI_GPT35_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT')
GPT35_API_VERSION = '2023-12-01-preview'
MODEL_35 = 'gpt-35-16k'

#The second model - gpt 4 mini
AZURE_OPENAI_GPT40_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT_4o')
GPT40_API_VERSION = '2024-08-01-preview'
MODEL_40 = 'gpt-4o-mini'


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

# Read the content of the text file
with open("tickets.txt", "r") as file:
    file_content = file.read()

# Use regex to extract each ticket
tickets = re.findall(r"\d+:\s(.*?)(?=\n\d+:|\Z)", file_content, re.DOTALL)
temperatures = [0, .5, .9]
models = [35, 40]
with open("categories.txt", "w") as file:
    for i, ticket in enumerate(tickets, 1):
        file.write(str(i) + ":\n")
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
            if model == 35:
                for temp in temperatures:
                    responses_gpt35 = gpt35_engine.chat.completions.create(messages=msg, model=MODEL_35, temperature=temp,
                                                                           max_tokens=MAX_TOKENS, logprobs=LOG_PROBS, n=1)

                    file.write(MODEL_35 + ", " + str(temp) + ": " + responses_gpt35.choices[0].message.content + "\n")
            else:
                for temp in temperatures:
                    responses_gpt40 = gpt40_engine.chat.completions.create(messages=msg, model=MODEL_40,
                                                                           temperature=temp,
                                                                           max_tokens=MAX_TOKENS, logprobs=LOG_PROBS,
                                                                           n=1)

                    file.write(MODEL_40 + ", " + str(temp) + ": " + responses_gpt40.choices[0].message.content + "\n")

file.close()






