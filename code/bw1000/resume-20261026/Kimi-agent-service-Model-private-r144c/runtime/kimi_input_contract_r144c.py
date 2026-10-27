"""Kimi's qualified Native bounds, checked before enqueue and before SSE."""
from native_engine_bridge_r143 import NativeEngineBridge


def validate_prompt_ids(ids, vocab_size):
    if not 1 <= len(ids) <= 512:
        raise ValueError("Kimi validated P0 prompt limit is 1..512 tokens; no truncation")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError("Kimi prompt token is outside the qualified Native vocabulary")


class KimiNativeEngineBridge(NativeEngineBridge):
    def _validate_native_output_limit(self, sampling_params):
        limit = getattr(sampling_params, "max_tokens", None)
        limit = self.default_max_tokens if limit is None else int(limit)
        if not 1 <= limit <= 512:
            raise ValueError("Kimi validated Native output budget is 1..512 tokens")

    def validate_data(self, data, sampling_params):
        self._validate_native_output_limit(sampling_params)
        return super().validate_data(data, sampling_params)

    def add_chat_request(self, *, messages, sampling_params, request_id,
                         request_data=None, add_generation_prompt=True,
                         chat_template_kwargs=None):
        self._validate_native_output_limit(sampling_params)
        return super().add_chat_request(messages=messages,
            sampling_params=sampling_params, request_id=request_id,
            request_data=request_data, add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs)
