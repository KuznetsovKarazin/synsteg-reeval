"""Bounded-v1 binding to the exact existing owned-proc Qwen service.

No model/tokenizer/process is started at import or construction. The old native
service preserves READY, owned PID, resident libraries and actual prompt checks;
the additional bounded context guard runs before it can start a native child.
"""
from .framed_interval_bounded_v1 import TOKEN_CAP, VERSION
from .framed_interval_llm_proc_service_v1 import ProcNativeService
from .framed_interval_llm_v1_journal import atomic_json


class BoundedNativeService(ProcNativeService):
    def __init__(self, activation, directory, deadline):
        super().__init__(activation, directory, deadline,
            adapter_readback=activation.data["source_readback"])

    def prepare(self):
        act = self.activation
        prompts = act.preflight["prompt_ids"]
        if set(prompts) != {"station", "library"}:
            raise ValueError("bounded prompt identities differ")
        if act.data["ready"]["context_tokens"] != 2048:
            raise ValueError("bounded context binding differs")
        for ids in prompts.values():
            if type(ids) is not list or len(ids) + TOKEN_CAP > 2048:
                raise ValueError("bounded prompt plus cap exceeds context before launch")
        endpoint = super().prepare()
        # Underlying resource_id deliberately remains the identity of the SAME
        # old native provider. Method domain/stop lives in the bounded envelope.
        atomic_json(self.directory / "bounded-native-binding.json", {
            "schema": VERSION + "-native-binding-v1", "status": "passed",
            "token_cap": TOKEN_CAP, "context_tokens": endpoint.context_tokens,
            "activation_sha256": act.sha256,
            "resource_id": endpoint.resource_id,
            "resource_sha256": endpoint.resource_digest,
            "provider_sha256": endpoint.provider_digest,
            "source_readback": act.data["source_readback"],
            "model_process_loaded": True, "prompt_calls_so_far": 0,
            "distribution_calls_so_far": 0, "step_calls_so_far": 0,
            "inherited_native_protocol_and_sampler_unchanged": True})
        return endpoint
