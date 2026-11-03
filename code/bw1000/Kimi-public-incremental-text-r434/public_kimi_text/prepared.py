"""Original qualified K3 codec/queue; no Native construction or HTTP injection."""
from public_kimi_native.runtime.long_input_r179 import KimiLongBridge, validate_prompt_ids
from public_kimi_native.public_history import make_prequeue_factory
from public_kimi_native.public_encode import encode
import threading


from .selector import TextDeliveryMixin, selection


class PublicKimiBridge(KimiLongBridge):
    def __init__(self, *args, **kwargs):
        self._admission_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def add_chat_request(self, **kwargs):
        with self._admission_lock:
            return super().add_chat_request(**kwargs)

    def stop(self):
        with self._admission_lock:
            super().stop()


class SelectedKimiBridge(TextDeliveryMixin, PublicKimiBridge):
    pass


def prepare(model_path, reasons, codec=None):
    if codec is None:
        from public_kimi_native.runtime.long_codec_r179 import KimiLongCodec
        codec = KimiLongCodec(model_path)

    def encode_chat(messages, data, generation, kwargs):
        prompt, ids = encode(codec, messages, data, generation, kwargs)
        validate_prompt_ids(ids, 163840)
        return prompt, ids

    history = make_prequeue_factory(codec)

    def prequeue(req, messages, data, generation, kwargs):
        # The original Kimi owner may dequeue immediately without acquiring
        # the HTTP admission lock. Bind selection before FIFO publication.
        bound = history(req, messages, data, generation, kwargs)
        bound['text_delivery'] = selection(data)
        return bound

    engine = SelectedKimiBridge(codec, 32768, reasons, encode_chat=encode_chat,
        decode_tokens=codec.decode, eos_token_ids={163585, 163586},
        default_max_tokens=4096, max_pending=8, prequeue_factory=prequeue)
    engine.validation_timeouts_s = dict(no_actual_progress=3600.0,
        absolute_request_including_queue=10800.0)
    return dict(engine=engine, codec=codec, FinishReason=reasons,
        identity=dict(codec.identity), parser='kimi_k3')
