from sae_lens import SAE
sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_20/width_16k/canonical")
m = sae.cfg.metadata
print("prepend_bos :", getattr(m, "prepend_bos", "?"))
print("context_size:", getattr(m, "context_size", "?"))
print("dataset_path:", getattr(m, "dataset_path", "?"))
print("--- full metadata dump (in case names differ) ---")
print(vars(m))
