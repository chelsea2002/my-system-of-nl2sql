from openai import OpenAI
from retrying import retry

client = OpenAI()

@retry(stop_max_attempt_number=3, wait_fixed=3000)
def LLM_generation(instruct, prompt, n=5):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instruct},
            {"role": "user", "content": prompt}
        ],
        n=n
    )
    results = [choice.message.content for choice in response.choices]
    return results