from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('CLASS_AZURE_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT')
OPENAI_API_VERSION = '2023-12-01-preview'
MODEL = 'gpt-35-16k'
TEMP = .3
MAX_TOKENS = 7000
LOG_PROBS = False

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

sys_prompt = {"role": "system", "content": "You are a dietition and a nutrition expert"}
user_prompt = {"role": "user", "content": "A restaurant wants to provide nutritional information \
for each dish it offers. what information should it provide? \
                                          limit your answer to no more than 10 requirements"}
msg = [sys_prompt, user_prompt]
responses = client.chat.completions.create(messages=msg, model=MODEL, temperature=TEMP,
                                           max_tokens=MAX_TOKENS, logprobs=LOG_PROBS, n=1)
print(responses.choices[0].message.content)

