"""NexteraBERT: a fast, long-context bidirectional encoder (SnowLily, sliding-window
and full attention, NexteraHRA)."""

# Exports are loaded LAZILY (PEP 562): importing the package - or one of its
# torch-free submodules like `nexterabert.streaming` - must NOT eagerly import
# `modeling_nexterabert`, which pulls in torch/CUDA. The pre-tokenisation script
# (`scripts/prepare_data.py`) depends on that: a pure-CPU tokenisation process that
# loads torch can crash at interpreter shutdown on some CUDA boxes
# ("PyGILState_Release: thread state must be current when releasing"). Attribute
# access (`nexterabert.NexteraBERT`, `from nexterabert import build_optimizer`)
# still works and imports the backing module on first use.
from importlib import import_module

__version__ = "0.1.0"

# exported name -> submodule that defines it
_EXPORTS = {
    "NexteraBERTConfig": "configuration_nexterabert",
    "PRESET_NAMES": "configuration_nexterabert",
    "NexteraBERTTokenizer": "tokenization_nexterabert",
    "push_to_hub": "export",
    "save_pretrained": "export",
    "save_mlm_head": "export",
    "load_for_task": "loading",
    "load_masked_lm": "loading",
    "build_optimizer": "optim",
    "MLP_CLASSES": "configuration_nexterabert",
    "NexteraBERT": "modeling_nexterabert",
    "Block": "modeling_nexterabert",
    "NexteraBERTForBertTrainer": "modeling_nexterabert",
    "NexteraBERTForCocoLM": "modeling_nexterabert",
    "NexteraBERTForCocoLMTrainer": "modeling_nexterabert",
    "NexteraBERTForDiscriminator": "modeling_nexterabert",
    "NexteraBERTForElectraTrainer": "modeling_nexterabert",
    "NexteraBERTForEmbedding": "modeling_nexterabert",
    "NexteraBERTForMaskedLM": "modeling_nexterabert",
    "NexteraBERTForMultipleChoice": "modeling_nexterabert",
    "NexteraBERTForQuestionAnswering": "modeling_nexterabert",
    "NexteraBERTForSequenceClassification": "modeling_nexterabert",
    "NexteraBERTForTokenClassification": "modeling_nexterabert",
    "DiscriminatorAccuracy": "modeling_nexterabert",
    "TokenAccuracy": "modeling_nexterabert",
    "is_ddp": "modeling_nexterabert",
    "get_dist_info": "modeling_nexterabert",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value          # cache so subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(__all__)
