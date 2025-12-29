import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# ============================================
# CONFIG
# ============================================

SENTENCE_ENCODER = "Alibaba-NLP/gte-base-en-v1.5"
HF_REPO = "mrfakename/ReverseBERT-GTE-Base-EN-1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# PROJECTION LAYER (must match training)
# ============================================

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * num_tokens),
        )
        self.output_dim = output_dim
    
    def forward(self, embeddings):
        projected = self.proj(embeddings)
        projected = projected.view(-1, self.num_tokens, self.output_dim)
        return projected

# ============================================
# LOAD MODELS
# ============================================

def load_models(
    sentence_encoder_name=SENTENCE_ENCODER,
    hf_repo=HF_REPO,
    device=DEVICE,
):
    from huggingface_hub import hf_hub_download
    
    print("Loading sentence encoder...")
    sentence_encoder = SentenceTransformer(sentence_encoder_name, device=device, trust_remote_code=True)
    sentence_encoder.eval()
    
    print(f"Loading LLM from {hf_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load the full model (includes LoRA merged or as adapter)
    llama = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    llama.eval()
    
    print("Loading projector...")
    # Download projector weights from HF repo
    projector_path = hf_hub_download(repo_id=hf_repo, filename="reverse_bert_projector.pt")
    
    projector = EmbeddingProjector(
        input_dim=768,
        output_dim=llama.config.hidden_size,
        num_tokens=4,
    ).to(device)
    projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
    projector.eval()
    
    return sentence_encoder, projector, llama, tokenizer

# ============================================
# INFERENCE
# ============================================

@torch.no_grad()
def reconstruct_text(
    text,
    sentence_encoder,
    projector,
    llama,
    tokenizer,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    device=DEVICE,
):
    """Reconstruct text from its sentence embedding."""
    # Encode text to embedding
    embedding = sentence_encoder.encode(text, convert_to_tensor=True).unsqueeze(0).to(device)
    
    # Project embedding to prefix tokens
    prefix_embeds = projector(embedding).to(torch.float16)
    
    # Generate text
    outputs = llama.generate(
        inputs_embeds=prefix_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@torch.no_grad()
def reconstruct_from_embedding(
    embedding,
    projector,
    llama,
    tokenizer,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    device=DEVICE,
):
    """Reconstruct text directly from a pre-computed embedding tensor."""
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.to(device)
    
    # Project embedding to prefix tokens
    prefix_embeds = projector(embedding).to(torch.float16)
    
    # Generate text
    outputs = llama.generate(
        inputs_embeds=prefix_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReverseBERT Inference")
    parser.add_argument("--text", type=str, help="Text to reconstruct")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    parser.add_argument("--repo", type=str, default=HF_REPO, help="HuggingFace repo ID")
    args = parser.parse_args()
    
    # Load models
    sentence_encoder, projector, llama, tokenizer = load_models(
        hf_repo=args.repo,
    )
    
    # Demo texts if none provided
    if args.text:
        test_sentences = [args.text]
    else:
        test_sentences = [
            "A man is speaking with a deep, gravelly voice.",
            "High pitched excited female voice with laughter.",
            "Cookie Monster speaks in a chaotic, muppet-like way.",
        ]
    
    print("\n" + "=" * 50)
    print("RECONSTRUCTION RESULTS")
    print("=" * 50)
    
    for sent in test_sentences:
        reconstructed = reconstruct_text(
            sent,
            sentence_encoder,
            projector,
            llama,
            tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )
        print(f"\nOriginal:      {sent}")
        print(f"Reconstructed: {reconstructed}")

