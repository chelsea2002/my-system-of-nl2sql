import os
from retrying import retry

_PROVIDER = os.getenv("LLM_PROVIDER", "remote").lower()

if _PROVIDER == "local":
    # 走本地
    from app.services.llm_local import LLM_generation as _local_generation

    @retry(stop_max_attempt_number=3, wait_fixed=3000)
    def LLM_generation(instruct, prompt, n=5, **kwargs):
        return _local_generation(instruct, prompt, n=n, **kwargs)

else:
    # 走远程（保留你原逻辑）
    from openai import OpenAI

    client = OpenAI()

    @retry(stop_max_attempt_number=3, wait_fixed=3000)
    def LLM_generation(instruct, prompt, n=5, **kwargs):
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": prompt},
            ],
            n=n,
        )
        return [choice.message.content for choice in response.choices]
