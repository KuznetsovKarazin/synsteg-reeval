"""framed-interval-llm-proc-service-v1: additive owned-procfs native binding.

No process is started at import. The frozen NativeBridge command methods and
fixed512 codec are inherited unchanged. Ownership is captured immediately after
Popen, before the first READY read; resident checks consume only that child's
maps. Construction of the service itself is side-effect free.
"""
import sys
import time
from pathlib import Path

from .framed_interval_llm_v1_activation import (
    RESOURCE_ROOT, safe_path, verify_resident_maps,
)
from .framed_interval_llm_v1_journal import atomic_json
from .framed_interval_llm_v1_orchestration import NativeService
from .framed_interval_llm_v1_preflight import chatml_prompt
from .framed_interval_llm_v1_qwen import prepare_qwen_endpoint
from .proc_namespace_identity_v2 import capture_owned_child
from .seeded_pair_llm import NativeBridge

VERSION = "framed-interval-llm-proc-service-v1"


class OwnedNativeBridge(NativeBridge):
    """Add ownership capture before the frozen constructor consumes READY."""
    def __init__(self, binary, model, cpu_backend, log_path, deadline, *,
                 executable_sha256, adapter_readback):
        self.identity_ticket = None
        self._capture_attempted = False
        self._capture_failed = False
        self._closed = False
        self._close_complete = False
        self._expected_binary = Path(binary)
        self._expected_binary_sha256 = executable_sha256
        self._adapter_readback = dict(adapter_readback)
        self._capture_path = Path(log_path).parent / "proc-identity-capture.json"
        try:
            super().__init__(binary, model, cpu_backend, log_path, deadline)
        except BaseException:
            # Frozen Popen is before its constructor's try. This also covers
            # launch failure after the stderr log was opened but before READY.
            self.close()
            raise

    def _line(self):
        if self._closed or self._capture_failed:
            raise ValueError("closed or failed owned bridge; no protocol retry")
        if not self._capture_attempted:
            self._capture_attempted = True
            self._capture_failed = True
            self.identity_ticket = capture_owned_child(self.process, self._expected_binary,
                                                       self._expected_binary_sha256)
            atomic_json(self._capture_path, {"schema": VERSION + "-capture",
                "popen_pid": self.process.pid, "parent": self.identity_ticket.parent,
                "child": self.identity_ticket.child,
                "executable": self.identity_ticket.expected_executable,
                "adapter_readback": self._adapter_readback})
            self._capture_failed = False
        return super()._line()

    def close(self):
        if self._close_complete: return
        self._closed = True
        original_error = sys.exception()
        errors = []
        try:
            super().close()
        except Exception as exc:
            errors.append(exc)
        # Base cleanup can fail before closing its streams. Attempt all owned
        # descriptors regardless; the parent worker still owns the process group.
        process = getattr(self, "process", None)
        for stream in (getattr(process, "stdin", None), getattr(process, "stdout", None),
                       getattr(self, "log", None)):
            if stream is not None:
                try: stream.close()
                except Exception as exc: errors.append(exc)
        if self.identity_ticket is not None:
            try: self.identity_ticket.close()
            except Exception as exc: errors.append(exc)
        self._close_complete = not errors
        if errors:
            detail = "; ".join(type(error).__name__ + ": " + str(error)[:256] for error in errors)
            if original_error is not None:
                original_error.add_note("Owned bridge cleanup: " + detail)
            else:
                raise RuntimeError("owned bridge cleanup failed: " + detail) from errors[0]


class ProcNativeService(NativeService):
    def __init__(self, activation, directory, deadline, *, adapter_readback):
        super().__init__(activation, directory, deadline)
        self.adapter_readback = dict(adapter_readback)

    def _resident(self, activation, bridge):
        ticket = bridge.identity_ticket
        if ticket is None or ticket.process is not bridge.process:
            raise ValueError("native bridge has no matching owned Popen ticket")
        maps, identity = ticket.maps_snapshot()
        result = verify_resident_maps(activation, bridge.process.pid, maps_text=maps)
        after = ticket.snapshot()
        if after != identity["after"]:
            raise ValueError("native identity changed during resident verification")
        return {**result, "proc_namespace_identity": identity,
                "post_resident_identity": after, "adapter_readback": self.adapter_readback}

    def prepare(self):
        # Same sequence/checks as frozen NativeService.prepare. Only construction
        # of the owned bridge and the resident lookup are additive replacements.
        act = self.activation
        if time.monotonic() >= self.deadline: raise TimeoutError("prelaunch deadline reached")
        resources = safe_path(act.root, RESOURCE_ROOT)
        self.process_attempted = True
        self.bridge = OwnedNativeBridge(resources / "seeded_pair_llama_bridge",
            resources / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            resources / "runtime/llama-b11791/libggml-cpu-haswell.so",
            self.directory / "native.stderr.log", self.deadline,
            executable_sha256=act.resource_report["verified_file_identities"]["seeded_pair_llama_bridge"]["sha256"],
            adapter_readback=self.adapter_readback)
        actual = {"vocabulary_size": self.bridge.vocabulary_size, "masked_tokens": self.bridge.masked_tokens,
                  "context_tokens": self.bridge.context_tokens}
        if actual != act.data["ready"]: raise ValueError("actual READY differs from activation")
        resident = self._resident(act, self.bridge)
        prompts = {}
        for prompt_id in ("station", "library"):
            before = getattr(self.bridge, "evaluated", None)
            ids = self.bridge.tokenize(chatml_prompt(act.public_protocol, prompt_id), True, True)
            if getattr(self.bridge, "evaluated", None) != before:
                raise ValueError("pre-PROMPT tokenizer command mutated model counter")
            if ids != act.preflight["prompt_ids"][prompt_id] or len(ids)+512 > 2048:
                raise ValueError("actual prompt IDs differ before PROMPT")
            prompts[prompt_id] = ids
        endpoint = prepare_qwen_endpoint(self.bridge, act.data["qwen_binding_manifest"],
                                         special_ids=frozenset(act.preflight["special_ids"]))
        atomic_json(self.directory / "native-preprompt-verification.json", {
            "schema": "framed-interval-llm-v1-native-preprompt-v1", "status": "passed",
            "ready": actual, "resident": resident, "prompt_ids": prompts,
            "activation_sha256": act.sha256, "prompt_calls_so_far": 0,
            "distribution_calls_so_far": 0, "step_calls_so_far": 0,
            "model_process_loaded": True, "no_claim_of_historical_system_byte_attestation": True})
        return endpoint
