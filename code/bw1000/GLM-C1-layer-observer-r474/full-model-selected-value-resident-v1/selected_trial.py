"""One resident Model, one transport, and explicit selected-value arms."""
from prefill_schedule import fragments

# The first pair diagnoses the 519-token prompt's singleton tail. The next
# three pairs use the original fragmentation and alternate the arm order.
PAIRS = ((23, ("off", "on")), (0, ("off", "on")),
         (17, ("on", "off")), (21, ("off", "on")))


def schedule():
    return [
        dict(index=pair * 2 + position, case_index=case, arm=arm,
             pair=pair, warmup=pair == 0, owner_cores=4)
        for pair, (case, order) in enumerate(PAIRS)
        for position, arm in enumerate(order)
    ]


def phases():
    return ["created", "loaded", *(f"case{row['index']}" for row in schedule()),
            "quiescent", "closed"]


def prompt_fragments(ids, singleton_tail):
    for offset, chunk in fragments(ids, 512):
        if singleton_tail and offset + len(chunk) == len(ids) and len(chunk) > 1:
            yield offset, chunk[:-1]
            yield len(ids) - 1, chunk[-1:]
        else:
            yield offset, chunk


class SelectedRequest:
    """Audit the counters on the real owner at boundaries outside decode timing."""

    def __init__(self, ctx, arm, prompt_tokens):
        if arm not in ("off", "on") or ctx.past != 0 or not 0 < prompt_tokens <= 28672:
            raise ValueError("invalid selected-value request start")
        self.ctx = ctx
        self.arm = arm
        self.prompt_tokens = prompt_tokens
        self.begin = ctx.selected_counters()
        self.prefill_end = None

    def forward(self, tokens, past, kind, *pointers):
        if kind == "prefill" and self.prefill_end is None and past + tokens <= self.prompt_tokens:
            packed = False
        elif kind == "decode" and self.prefill_end is not None and past >= self.prompt_tokens and tokens == 1:
            packed = self.arm == "on"
        else:
            raise ValueError("forward does not belong to the declared prompt or continuation")
        self.ctx.forward(tokens, past, *pointers, phase=kind, packed=packed)

    def finish_prefill(self):
        if self.ctx.past != self.prompt_tokens or self.prefill_end is not None:
            raise ValueError("incomplete or repeated prefill")
        observed = self.ctx.selected_counters()
        if self.begin != observed or observed[0] != 1:
            raise ValueError("prefill modified selected-value counters")
        self.prefill_end = observed

    def report(self, decode_calls):
        if self.prefill_end is None or self.ctx.past != self.prompt_tokens + decode_calls:
            raise ValueError("continuation count differs from the owner cursor")
        end = self.ctx.selected_counters()
        packed = end[1] - self.prefill_end[1]
        cached = end[2] - self.prefill_end[2]
        if end[0] != 1 or not 0 <= cached <= packed <= decode_calls or (self.arm == "off" and packed):
            raise ValueError("invalid selected-value continuation counters")
        return dict(begin=self.begin, after_prefill=self.prefill_end, end=end,
                    packed_calls=packed, cached_dispatch_calls=cached,
                    packed_path_observed=packed > 0)
