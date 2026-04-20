import os
from typing import Dict, List, Optional


class AzureOpenAIWrapper:
    """Azure OpenAI chat wrapper.

    The rest of this codebase expects an "agent" object. Local agents expose
    `huggingface_model` + `tokenizer`. For API agents we expose:

      - kind = "azure_openai"
      - model_name: str
      - complete(messages: List[Dict[str, str]]) -> str

    This keeps the integration minimal and allows heterogeneous agent lists.
    """

    kind = "azure_openai"

    def __init__(
        self,
        model_name: str,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2025-01-01-preview",
        timeout: Optional[float] = None,
    ):
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.timeout = timeout

        try:
            # openai>=1.0 provides AzureOpenAI
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "AzureOpenAIWrapper requires the official OpenAI Python SDK. "
                "Install with: pip install 'openai>=1.0.0'"
            ) from e

        self._client = OpenAI(
            base_url=self.azure_endpoint,
            api_key=self.api_key,
            #api_version=self.api_version,
        )
        
    def chat_completion(self, messages, max_tokens, temperature, top_p, **kwargs):
        if self.model_name == "o3-mini":
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                #top_p=top_p,
                **kwargs
            )
        else:
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Call Azure Chat Completions.

        `messages` follows the OpenAI chat format:
          [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
        """
        MAX_RETRY_TOKENS = 4096

        max_tokens_try = max_tokens
        while max_tokens_try <= MAX_RETRY_TOKENS:
            try:
                resp = self.chat_completion(messages, max_tokens_try, temperature, top_p, **kwargs)
                break
            except BadRequestError as e:
                if "Insufficient tokens to fulfill request" in str(e): #not enough tokens
                    max_tokens_try *= 2
                    if max_tokens_try > MAX_RETRY_TOKENS:
                        raise ValueError(
                            f"Request costs more than token limit at {MAX_RETRY_TOKENS}"
                        ) from e
                else:
                    raise

        return resp.choices[0].message.content or ""
