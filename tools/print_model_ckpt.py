from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./checkpoints/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("=" * 80)
print("MODEL STRUCTURE")
print("=" * 80)
print(model)
print("\n")

print("=" * 80)
print("MODEL PARAMETERS")
print("=" * 80)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
print("\n")

print("=" * 80)
print("MODEL LAYERS")
print("=" * 80)
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")
print("\n")

print("=" * 80)
print("TOKENIZER STRUCTURE")
print("=" * 80)
print(tokenizer)
print("\n")

print("=" * 80)
print("TOKENIZER CONFIG")
print("=" * 80)
print(f"Vocabulary size: {len(tokenizer)}")
print(f"Max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"Eos token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"Bos token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"Unk token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
print(f"Mask token: {tokenizer.mask_token} (id: {tokenizer.mask_token_id})")
print("\n")

print("=" * 80)
print("SPECIAL TOKENS")
print("=" * 80)
print(tokenizer.special_tokens_map)
print("\n")

print("=" * 80)
print("TOKENIZER VOCABULARY (First 20 tokens)")
print("=" * 80)
vocab = tokenizer.get_vocab()
for i, (token, idx) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:20]):
    print(f"{idx}: {token}")
print("\n")

print("=" * 80)
print("MODEL SUMMARY")
print("=" * 80)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {total_params - trainable_params:,}")
print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")