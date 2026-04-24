from typing import Dict, List


class AttentionTracer:
    """Trace and store attention patterns step-by-step during episode execution."""

    def __init__(self):
        self.traces: List[Dict] = []

    def add_trace(self, trace: Dict):
        self.traces.append(trace)

    def get_traces(self) -> List[Dict]:
        return self.traces

    def visualize_attention(self, step: int):
        """Visualize attention at a specific step (extensible)."""
        if step < len(self.traces):
            _trace = self.traces[step]
            # extend here: plot dec_attn / enc_attn fields from the trace dict
            pass
