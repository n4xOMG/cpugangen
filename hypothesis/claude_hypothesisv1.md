# Research Hypothesis (Cải tiến): Hybrid Lightweight Text Conditioning

> **Tên giả thuyết:** "Hybrid Semantic Encoding for CPU-Efficient Anime Generation"
> 
> **Phiên bản:** v2.0 (Cải tiến từ "Embedding Lookup Hypothesis")
> 
> **Ngày tạo:** 2026-01-01
> 
> **Mục tiêu:** Cân bằng tốc độ, RAM và chất lượng cho gen ảnh Anime trên CPU

---

## 📋 Tóm tắt Phản biện Giả thuyết v1.0

### Vấn đề chính đã phát hiện:

1. ❌ **Bottleneck Fallacy:** CLIP chỉ chiếm ~5% tổng thời gian (Denoising Loop mới là nút thắt)
2. ❌ **Attribute Bleeding:** Average pooling làm mất thông tin tag interactions
3. ❌ **Cold Start Problem:** Random init embedding rất khó converge
4. ❌ **OOV Crisis:** 5000 tags không đủ cho long-tail character/style tags

---

## 🎯 Giả thuyết Mới (H2)

### Phát biểu:

> Một **Hybrid Text Encoder** kết hợp **Distilled CLIP** (cho semantic understanding) và **Learnable Tag Embeddings** (cho specific anime concepts) sẽ đạt được:
> 
> 1. **Tốc độ:** Inference nhanh hơn 40-60% so với CLIP ViT-B/32 full
> 2. **RAM:** Giảm 50-70% memory footprint
> 3. **Chất lượng:** Duy trì FID score trong ±3% so với baseline
> 4. **Flexibility:** Xử lý được cả common tags và rare character names

---

## 🏗️ Kiến trúc Đề xuất

### Architecture Overview:

```
Input Tags: ["hatsune_miku", "twin_tails", "sitting"]
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
[Common Path]    [Rare Path]      [Semantic Path]
Tag Embedding    Character DB     Distilled CLIP
(~3K tags)       (Lookup Table)   (TinyBERT)
    │                 │                 │
    │                 │                 │
    └─────────────────┴─────────────────┘
                      │
                      ▼
              [Fusion Module]
           (Lightweight Attention)
                      │
                      ▼
             768-dim Conditioning
                      │
                      ▼
                  DiT-Tiny
```

---

## 🔧 Implementation Details

### 1. Three-Branch Encoder

```python
class HybridTextEncoder(nn.Module):
    def __init__(self, 
                 common_vocab_size=3000,    # Top 3000 frequent tags
                 character_db_size=5000,     # Character names lookup
                 embed_dim=768):
        super().__init__()
        
        # Branch 1: Common Tags (trainable)
        self.common_embedding = nn.Embedding(common_vocab_size, 256)
        
        # Branch 2: Character Database (frozen, pre-computed)
        self.character_db = self._load_character_embeddings()  # From CLIP
        
        # Branch 3: Distilled CLIP (frozen or light fine-tune)
        self.semantic_encoder = TinyBERT.from_pretrained('distilbert-base')
        self.semantic_proj = nn.Linear(384, 256)  # TinyBERT outputs 384-dim
        
        # Fusion: Lightweight cross-attention
        self.fusion = CrossAttentionFusion(
            input_dim=256,
            output_dim=embed_dim,
            num_heads=4
        )
    
    def forward(self, tags: List[str]):
        # Classify tags into 3 categories
        common_tags, character_tags, semantic_tags = self._classify_tags(tags)
        
        # Branch 1: Common tags lookup
        common_embeds = self.common_embedding(common_tags)  # [N_common, 256]
        
        # Branch 2: Character exact match
        char_embeds = self.character_db[character_tags]     # [N_char, 256]
        
        # Branch 3: Semantic encoding for rare/compositional tags
        semantic_embeds = self.semantic_proj(
            self.semantic_encoder(semantic_tags)
        )  # [N_semantic, 256]
        
        # Concatenate all
        all_embeds = torch.cat([common_embeds, char_embeds, semantic_embeds], dim=0)
        
        # Fusion with attention (learns tag importance)
        fused = self.fusion(all_embeds)  # [768]
        
        return fused
```

### 2. Fusion Module (Thay thế Average Pooling)

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim=256, output_dim=768, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, output_dim))  # Learnable query
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        
    def forward(self, tag_embeds):
        # tag_embeds: [num_tags, 256]
        keys = self.key_proj(tag_embeds).unsqueeze(0)    # [1, num_tags, 768]
        values = self.value_proj(tag_embeds).unsqueeze(0)
        
        # Query attends to all tags
        out, attn_weights = self.attn(
            self.query.unsqueeze(0),  # [1, 1, 768]
            keys, 
            values
        )
        return out.squeeze(0)  # [768]
```

### 3. Character Database Pre-computation

```python
# One-time preprocessing
def build_character_db():
    """
    Extract top 5000 character names from Danbooru
    Pre-compute their CLIP embeddings
    """
    clip_model = load_clip_model()
    character_names = load_danbooru_characters(top_k=5000)
    
    db = {}
    for name in character_names:
        with torch.no_grad():
            embedding = clip_model.encode_text(name)  # [512]
            # Project to 256-dim
            embedding = nn.Linear(512, 256)(embedding)
        db[name] = embedding
    
    torch.save(db, 'character_embeddings.pt')
    return db
```

---

## 📊 Thiết kế Thí nghiệm

### Baseline Models:

| Model | Text Encoder | Params | RAM | Inference Time |
|-------|--------------|--------|-----|----------------|
| **A (Full CLIP)** | CLIP ViT-B/32 | 150M | 600MB | 100% (baseline) |
| **B (Distilled)** | TinyBERT only | 66M | 250MB | 60% |
| **C (Hybrid - Ours)** | 3-Branch Hybrid | 70M | 280MB | 50-60% |
| **D (Embedding)** | Pure Embedding Table | 3.8M | 15MB | 10% (but low quality) |

### Dataset:

- **Training:** 100,000 ảnh Danbooru2021
- **Validation:** 10,000 ảnh
- **Test Set:** 5,000 ảnh với:
  - 2000 common tags only
  - 2000 character-specific prompts (hatsune_miku, etc.)
  - 1000 rare/compositional prompts

### Metrics Toàn diện:

| Metric | Tool | Target |
|--------|------|--------|
| **FID Score** | pytorch-fid | Model C ≤ Model A + 3% |
| **CLIP Score** | CLIP model | Model C ≥ 0.90 × Model A |
| **Tag Accuracy** | WD-Tagger v2 | Top-5 accuracy ≥ 85% |
| **Attribute Separation** | Custom (đo overlap giữa red_hair/blue_eyes) | < 10% bleeding |
| **Character Fidelity** | LPIPS + Human eval | ≥ 90% recognizable |
| **Inference Time** | `time.perf_counter()` | 40-60% faster |
| **RAM Peak** | `psutil` | ≤ 280MB |

---

## 🔬 Ablation Studies

### Experiment 1: Branch Importance

Train 3 variants:
- **H2a:** Common + Character only (no semantic)
- **H2b:** Common + Semantic only (no character DB)
- **H2c:** Full 3-branch (proposed)

**Hypothesis:** H2c sẽ tốt nhất trên test set đa dạng

### Experiment 2: Fusion Mechanisms

Compare:
- Average Pooling (baseline v1.0)
- Max Pooling
- Self-Attention Pooling
- **Cross-Attention Pooling (ours)**

**Hypothesis:** Cross-attention giảm attribute bleeding xuống < 5%

### Experiment 3: Character DB Size

Vary character_db_size: [0, 1000, 5000, 10000]

**Hypothesis:** 5000 là sweet spot (coverage vs memory)

---

## 🎯 Giải quyết Phản biện

### ✅ Bottleneck Fallacy

**Solution:** Không tập trung vào loại bỏ CLIP hoàn toàn, mà làm nó nhẹ hơn (TinyBERT) + cache character embeddings

**Impact:** Giảm text encoding từ 1s → 0.4s (vẫn có ý nghĩa khi user generate nhiều ảnh)

### ✅ Attribute Bleeding

**Solution:** Cross-Attention Fusion thay vì average pooling

**Proof:** Attention weights sẽ học được "blue" nên áp dụng cho "hair" hay "eyes"

### ✅ Cold Start

**Solution:** 
- Common embeddings: Initialize từ CLIP text encoder
- Character DB: Pre-computed từ CLIP (frozen)
- TinyBERT: Pretrained on 1M text corpus

**Impact:** Model converges trong 20k iterations thay vì 100k+

### ✅ OOV Problem

**Solution:** 3-tier system:
1. Common tags → Embedding lookup (fast)
2. Known characters → Database lookup (exact match)
3. Rare/Unknown → TinyBERT semantic encoding (fallback)

**Coverage:** ~95% tags được xử lý tốt (thay vì 60-70% với 5000-only)

---

## ⚠️ Rủi ro Mới

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Complexity overhead | Medium | Minimize branches (remove semantic if needed) |
| Character DB maintenance | Low | Auto-update từ Danbooru API |
| Attention computation slow | Medium | Use linear attention (Linformer) thay vì full |
| Training instability | Low | Freeze TinyBERT, chỉ train fusion + common embeddings |

---

## 📅 Timeline Thực tế

### Week 1-2: Preparation
- [_] Build character database (5000 names)
- [_] Pre-compute CLIP embeddings
- [_] Setup TinyBERT distillation pipeline
- [_] Prepare dataset splits with tag classification

### Week 3-4: Implementation
- [_] Implement HybridTextEncoder
- [_] Implement CrossAttentionFusion
- [_] Integrate with DiT-Tiny
- [_] Setup training loop with mixed precision

### Week 5-8: Training (Realistic CPU timeline)
- [_] Train Model A (CLIP baseline) - 2 weeks
- [_] Train Model C (Hybrid) - 2 weeks
- [_] Run ablation studies

### Week 9-10: Evaluation
- [_] Calculate all metrics (FID, CLIP Score, etc.)
- [_] Human evaluation (50 prompts × 4 models)
- [_] Statistical significance tests

### Week 11-12: Analysis
- [_] Ablation result analysis
- [_] Error case study
- [_] Write final report

**Total:** 12 weeks (3 months) - Thực tế hơn nhiều so với 4 tuần ban đầu

---

## 🏆 Kết quả Kỳ vọng

### Best Case (H2 accepted):

- **Speed:** 50% faster than CLIP full (từ 20s → 10s per image)
- **Quality:** FID trong ±2% so với baseline
- **Character accuracy:** 95%+ on known characters
- **Contribution:** Practical solution cho CPU users

### Worst Case (H2 rejected):

**Learning:** Semantic understanding không thể distill xuống mô hình nhỏ hơn mà không mất quality

**Pivot:** Quay lại dùng CLIP full nhưng optimize inference (ONNX, quantization, caching)

---

## 💡 Follow-up Research Directions

Nếu H2 thành công, có thể mở rộng:

1. **Dynamic Branch Selection:** Tự động chọn branch nào chạy dựa trên input tags (save compute)
2. **Progressive Loading:** Load character DB on-demand thay vì toàn bộ vào RAM
3. **Multi-modal Character DB:** Thêm visual embeddings cho characters (CLIP image encoder)
4. **User-customizable Character DB:** Cho phép user thêm custom characters

---

## 📝 Câu hỏi Mở

1. **Architecture:**
   - Liệu 3 branches có quá phức tạp? Có thể giảm xuống 2?
   - Cross-attention có thật sự cần thiết hay self-attention đủ?

2. **Training:**
   - Nên freeze TinyBERT hay fine-tune?
   - Learning rate schedule như thế nào cho multi-branch?

3. **Evaluation:**
   - Human evaluation cần bao nhiêu samples để đạt statistical significance?
   - Có cần thêm artist-specific metrics không (line quality, color harmony)?

4. **Deployment:**
   - Character DB 5000 × 256 float32 = ~5MB, có chấp nhận được cho mobile không?
   - Có thể quantize embeddings xuống int8 mà không mất accuracy?

---

## 🎓 Tài liệu Tham khảo

### Papers:
- DistilBERT (Sanh et al., 2019)
- CLIP (Radford et al., 2021)
- DiT (Peebles & Xie, 2023)
- Linformer (Wang et al., 2020) - Linear Attention

### Datasets:
- Danbooru2021 (https://www.gwern.net/Danbooru2021)
- AnimeFace Character Dataset

### Tools:
- Hugging Face Transformers (TinyBERT)
- pytorch-fid (FID calculation)
- WD-Tagger v2 (Tag accuracy)

---

> **Tóm tắt cải tiến so với v1.0:**
> 
> | Aspect | v1.0 (Embedding Only) | v2.0 (Hybrid) |
> |--------|----------------------|---------------|
> | Architecture | Single embedding table | 3-branch hybrid |
> | Pooling | Average (naive) | Cross-attention |
> | Initialization | Random | Distilled from CLIP |
> | OOV handling | [UNK] token | Semantic fallback |
> | Expected quality | Low (attribute bleeding) | High (preserved semantic) |
> | Timeline | 4 weeks (unrealistic) | 12 weeks (realistic) |
> | Success probability | ~30% | ~70% |

---

**Recommendation:** Tiến hành thí nghiệm với giả thuyết H2 (Hybrid). Nếu cần cắt giảm scope, có thể bỏ semantic branch và chỉ giữ Common + Character (2-branch variant).
