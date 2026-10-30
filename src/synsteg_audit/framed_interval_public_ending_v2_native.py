"""Ending-v2 adapter reuses frozen native resources/profile; no calls at import."""
from dataclasses import replace
from pathlib import Path
import time
from .canonical_channel_v2 import CanonicalGate, ContractError
from .framed_interval_llm_v1 import Endpoint
from .framed_interval_llm_v1_journal import atomic_json
from .framed_interval_llm_v1_qwen import prepare_qwen_endpoint
from .framed_interval_public_v1_native import (
    validate_profile, resource_descriptor, verify_files, verify_resident,
    PublicOwnedBridge, RESOURCE_ROOT, BRIDGE, MODEL, CPU,
)
from .framed_interval_public_ending_v2 import (
    VERSION, public_digest, validate_public_context, validate_public_spec,
)

class PublicNativeService:
    kind="native"
    def __init__(self, root, profile, public_spec, public_context, directory, deadline):
        self.root=Path(root); self.profile=profile; self.spec=public_spec; self.context=public_context
        self.directory=Path(directory);self.deadline=deadline;self.bridge=None;self.process_attempted=False;self.prepare_attempted=False

    def prepare(self):
        if self.prepare_attempted: raise ContractError("native service is one-attempt; no retry")
        self.prepare_attempted=True
        validate_profile(self.profile)
        validate_public_spec(self.spec)
        ids=validate_public_context(self.context,self.spec)
        if self.spec["resources"]!=resource_descriptor(self.profile):
            raise ContractError("spec does not bind this complete public profile")
        if any(t>=self.profile["ready"]["vocabulary_size"] for t in ids):
            raise ContractError("public context has out-of-vocabulary ID")
        resource_check=verify_files(self.root,self.profile)
        atomic_json(self.directory/"native-resource-verification.json",resource_check)
        if time.monotonic()>=self.deadline:raise TimeoutError("prelaunch deadline reached")
        base=self.root/RESOURCE_ROOT
        self.process_attempted=True
        self.bridge=PublicOwnedBridge(base/BRIDGE,base/MODEL,base/CPU,self.directory/"native.stderr.log",self.deadline,
            expected_sha256=self.profile["resource_files"][BRIDGE]["sha256"],profile_sha256=public_digest(self.profile))
        ready={"vocabulary_size":self.bridge.vocabulary_size,"masked_tokens":self.bridge.masked_tokens,
            "context_tokens":self.bridge.context_tokens}
        if ready!=self.profile["ready"]:raise ContractError("actual READY differs")
        resident=verify_resident(self.root,self.profile,self.bridge)
        old=prepare_qwen_endpoint(self.bridge,self.profile["qwen_binding"],special_ids=frozenset(self.profile["special_ids"]))
        desc=resource_descriptor(self.profile)
        codec=replace(old.gate.codec,resource_digest=desc["codec_sha256"])
        endpoint=Endpoint(CanonicalGate(codec),resource_id=desc["resource_id"],resource_digest=desc["resource_sha256"],
            provider_digest=desc["provider_sha256"],distribution_callback=old._distribution,
            start_callback=old._start,observe_callback=old._observe,context_tokens=2048)
        validate_public_spec(self.spec,endpoint)
        atomic_json(self.directory/"native-preprompt-verification.json",{
            "schema":VERSION+"-native-preprompt","status":"passed","ready":ready,"resident":resident,
            "public_profile_sha256":public_digest(self.profile),"public_spec_sha256":public_digest(self.spec),
            "public_context_sha256":public_digest(self.context),"prompt_ids":list(ids),
            "context_token_count":len(ids),"model_process_loaded":True,"prompt_calls_so_far":0,
            "distribution_calls_so_far":0,"step_calls_so_far":0,"tokenizer_preflight_repeated":False})
        return endpoint

    def close(self):
        if self.bridge is not None:self.bridge.close()
