"""LLM clients.
  OpenRouterLLM - real model through OpenRouter's OpenAI-compatible API (one key, many models)
  ScriptedLLM   - deterministic stand-in for tests: replays a script of responses

Both return the same dict: {content, tool_calls: [{id, name, arguments}], usage: {...}}."""
import json
import os
import time

from .. import config


class OpenRouterLLM:
    def __init__(self, model=None, temperature=0.0, max_tokens=2000, seed=None):
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in skeptic/.env)")
        self.client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=key)
        self.model = model or config.MODEL
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed

    def chat(self, messages, tools=None, tool_choice=None):
        """One chat-completion call (retried on transient API errors). Returns {content, tool_calls, usage};
        tool arguments that are not valid JSON come back as {'_unparseable': text}."""
        kw = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
                  extra_body={"usage": {"include": True}})          # OpenRouter returns the cost in usage
        if self.seed is not None:
            kw["seed"] = self.seed
        if tools:
            kw["tools"] = tools
            if tool_choice:
                kw["tool_choice"] = tool_choice
        for attempt in range(4):                                     # retry transient API errors
            try:
                r = self.client.chat.completions.create(**kw)
                break
            except Exception:                                   # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            calls.append(dict(id=tc.id, name=tc.function.name, arguments=args))
        u = r.usage
        usage = dict(prompt_tokens=getattr(u, "prompt_tokens", 0), completion_tokens=getattr(u, "completion_tokens", 0),
                     cost_usd=float(getattr(u, "cost", 0.0) or 0.0))
        return dict(content=msg.content or "", tool_calls=calls, usage=usage)


class ScriptedLLM:
    """script: list of responses, each either a dict {content, tool_calls} or a callable
    (messages) -> dict. tool_calls may omit ids. Runs out -> returns plain text."""

    def __init__(self, script, model="scripted"):
        self.script, self.i, self.model = list(script), 0, model
        self.seen = []

    def chat(self, messages, tools=None, tool_choice=None):
        """Next scripted reply (a dict, or a function of the messages), in the same format as OpenRouterLLM.chat."""
        self.seen.append(messages)
        if self.i >= len(self.script):
            return dict(content="(script exhausted)", tool_calls=[], usage=dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0))
        r = self.script[self.i]
        self.i += 1
        r = r(messages) if callable(r) else r
        calls = [dict(id=c.get("id", f"s{self.i}_{k}"), name=c["name"], arguments=c.get("arguments", {}))
                 for k, c in enumerate(r.get("tool_calls", []))]
        return dict(content=r.get("content", ""), tool_calls=calls,
                    usage=dict(prompt_tokens=100, completion_tokens=20, cost_usd=0.0))
