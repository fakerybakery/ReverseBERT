import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

SENTENCE_ENCODER = "Alibaba-NLP/gte-base-en-v1.5"  # 768 dim
LLAMA_MODEL = "Qwen/Qwen3-0.6B-Base"
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
LR = 2e-4
MAX_SEQ_LEN = 512
DEVICE = "cuda"

# ============================================
# LOAD MODELS
# ============================================

print("Loading sentence encoder...")
sentence_encoder = SentenceTransformer(SENTENCE_ENCODER, device=DEVICE, trust_remote_code=True)
sentence_encoder.eval()
for p in sentence_encoder.parameters():
    p.requires_grad = False

print("Loading LLaMA...")
# bnb_config = BitsAndBytesConfig(
#     # load_in_8bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

llama = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    # quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# add LoRA
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

llama = get_peft_model(llama, lora_config)
llama.print_trainable_parameters()

# ============================================
# PROJECTION LAYER
# embedding dim (768) -> llama hidden dim
# ============================================

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        # project to multiple "soft prompt" tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * num_tokens),
        )
        self.output_dim = output_dim
    
    def forward(self, embeddings):
        # embeddings: (batch, 768)
        projected = self.proj(embeddings)  # (batch, output_dim * num_tokens)
        projected = projected.view(-1, self.num_tokens, self.output_dim)  # (batch, num_tokens, output_dim)
        return projected

projector = EmbeddingProjector(
    input_dim=768,
    output_dim=llama.config.hidden_size,
    num_tokens=4,  # how many "prefix" tokens the embedding becomes
).to(DEVICE, dtype=torch.float16)  # match LLM dtype

# ============================================
# DATASET
# ============================================

class TextReconstructionDataset(Dataset):
    def __init__(self, texts, sentence_encoder, tokenizer, max_len=128):
        self.texts = texts
        self.sentence_encoder = sentence_encoder
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # get sentence embedding
        with torch.no_grad():
            embedding = self.sentence_encoder.encode(
                text, 
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        
        # tokenize target text
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "embedding": embedding,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }

# load some data - using a simple dataset for demo
# you could swap this for audio captions
print("Loading dataset...")
dataset = load_dataset("mrfakename/voice-acting-inst-en", split="train")
texts = [ex["instruction"] for ex in dataset]  # truncate long reviews

train_dataset = TextReconstructionDataset(
    texts=texts,
    sentence_encoder=sentence_encoder,
    tokenizer=tokenizer,
    max_len=MAX_SEQ_LEN,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# ============================================
# TRAINING LOOP
# ============================================

optimizer = torch.optim.AdamW(
    list(projector.parameters()) + list(llama.parameters()),
    lr=LR,
)

llama.train()
projector.train()

for epoch in range(EPOCHS):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in enumerate(pbar):
        embeddings = batch["embedding"].to(DEVICE, dtype=torch.float16)  # match LLM dtype
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        
        # project embeddings to prefix tokens
        prefix_embeds = projector(embeddings)  # (batch, num_tokens, hidden_dim)
        
        # get token embeddings for the target text
        token_embeds = llama.get_input_embeddings()(input_ids)  # (batch, seq_len, hidden_dim)
        
        # concat: [prefix_embeds, token_embeds]
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        # extend attention mask for prefix tokens
        prefix_mask = torch.ones(
            embeddings.size(0), 
            prefix_embeds.size(1),
            device=DEVICE,
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # create labels: -100 for prefix (don't compute loss), then input_ids
        prefix_labels = torch.full(
            (embeddings.size(0), prefix_embeds.size(1)),
            -100,
            device=DEVICE,
            dtype=input_ids.dtype,
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)
        
        # forward pass
        outputs = llama(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# ============================================
# SAVE
# ============================================

print("Saving...")
llama.save_pretrained("./reverse_bert_llama")
llama.push_to_hub("ReverseBERT")
torch.save(projector.state_dict(), "./reverse_bert_projector.pt")

# ============================================
# INFERENCE / TEST
# ============================================

@torch.no_grad()
def reconstruct_text(text, sentence_encoder, projector, llama, tokenizer, max_new_tokens=64):
    llama.eval()
    projector.eval()
    
    # encode
    embedding = sentence_encoder.encode(text, convert_to_tensor=True).unsqueeze(0).to(DEVICE, dtype=torch.float16)
    
    # project
    prefix_embeds = projector(embedding)
    
    # generate
    generated_ids = []
    current_embeds = prefix_embeds
    
    for _ in range(max_new_tokens):
        outputs = llama(inputs_embeds=current_embeds)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        generated_ids.append(next_token.item())
        next_embed = llama.get_input_embeddings()(next_token).unsqueeze(1)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# test it
print("\n" + "="*50)
print("TESTING RECONSTRUCTION")
print("="*50)

test_sentences = [
    "A man is speaking with a deep, gravelly voice.",
    "High pitched excited female voice with laughter.",
    "Cookie Monster speaks in a chaotic, muppet-like way.",
]

for sent in test_sentences:
    reconstructed = reconstruct_text(
        sent, sentence_encoder, projector, llama, tokenizer
    )
    print(f"\nOriginal:      {sent}")
    print(f"Reconstructed: {reconstructed}")