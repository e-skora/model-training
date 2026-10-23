"""Load the base model in 4-bit, optionally with a trained adapter, ready to answer.

GPU only: the imports happen inside the function, so importing this file
needs no GPU libraries. Used by `scripts/train.py` (validation answers) and
`scripts/eval_trained.py` (test answers), so both load the model the same way.
"""

MAX_SEQ_LENGTH = 2048  # the same as the baseline runs


def load_judge(repo, revision, adapter=None, adapter_revision=None):
    """Return (model, tokenizer). `adapter` is a local folder or a Hugging Face repo with a LoRA adapter."""
    import unsloth  # noqa: F401  (import first, as Unsloth asks)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, revision=revision
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, revision=adapter_revision, is_trainable=False)
    FastLanguageModel.for_inference(model)
    return model, tokenizer
