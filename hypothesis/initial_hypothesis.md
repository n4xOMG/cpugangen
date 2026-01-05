# Research Hypothesis: Embedding Lookup for CPU-based Anime Generation

> **Tên giả thuyết:** "Tra bảng nhanh hơn suy luận" (The Embedding Lookup Hypothesis)
> 
> **Ngày tạo:** 2026-01-01
> 
> **Mục tiêu:** Tối ưu tốc độ sinh ảnh Anime trên CPU

---

## 1. Bối cảnh nghiên cứu

- **Phần cứng mục tiêu:** CPU thông thường (không có GPU/Tensor cores)
- **Dữ liệu:** Danbooru tags (từ khóa rời rạc, không phải câu văn)
- **Ưu tiên:** Tốc độ inference > Chất lượng tuyệt đối

---

## 2. Phát biểu giả thuyết

### Giả thuyết $H_1$:

> Đối với dữ liệu đầu vào dạng từ khóa (tags) cố định như Danbooru, việc **thay thế mô hình ngôn ngữ (Text Encoder như CLIP)** bằng một **lớp Trainable Embedding Table** (Bảng tra cứu vector có thể học) sẽ:
> 
> 1. Giảm thời gian tiền xử lý (preprocessing time) xuống **gần bằng 0**
> 2. Giảm **20-30% RAM** tiêu thụ
> 3. **Không làm giảm chất lượng ảnh** (FID score) trong phạm vi dữ liệu Anime

### Giả thuyết $H_0$ (Null Hypothesis):

> Việc thay thế CLIP bằng Embedding Table sẽ làm giảm đáng kể chất lượng ảnh sinh ra do mất đi khả năng hiểu ngữ nghĩa của text encoder.

---

## 3. Cơ sở lý luận

### 3.1 Tại sao CLIP là "overkill" cho Danbooru?

| Đặc điểm | CLIP được thiết kế cho | Danbooru tags thực tế |
|----------|------------------------|----------------------|
| Cấu trúc | Câu văn phức tạp: *"A cat sitting on a red carpet near the window"* | Danh sách từ rời rạc: `1girl, cat_ears, sitting, red_carpet` |
| Ngữ pháp | Hiểu quan hệ ngữ pháp (chủ-vị-tân) | Không có ngữ pháp, chỉ là tập hợp |
| Từ vựng | Từ vựng mở (open vocabulary) | Từ vựng đóng (~10,000 tags phổ biến) |
| Tính toán | Transformer với hàng trăm triệu tham số | Chỉ cần mapping tag → vector |

### 3.2 Độ phức tạp tính toán

```
CLIP ViT-B/32:
├── Tham số: ~150 triệu
├── Thao tác: Matrix multiplication, Self-Attention
└── Độ phức tạp: O(n²) cho sequence length n

Embedding Table:
├── Tham số: vocab_size × embed_dim = 5000 × 768 = ~3.8 triệu
├── Thao tác: Index lookup + Average pooling
└── Độ phức tạp: O(1) cho mỗi tag lookup
```

### 3.3 Đặc thù của ảnh Anime

- **Phong cách nhất quán:** Anime có conventions rõ ràng (mắt to, tóc màu sắc, v.v.)
- **Tags có tính deterministic:** `blue_eyes` luôn có nghĩa như nhau, không cần "hiểu ngữ cảnh"
- **Composition patterns lặp lại:** Các tổ hợp tag phổ biến xuất hiện nhiều lần trong dataset

---

## 4. Thiết kế thí nghiệm

### 4.1 Nhóm đối chứng (Control Group)

```python
# Model A: Baseline với CLIP
model_a = DiT_Tiny(
    text_encoder=CLIP_ViT_B32(freeze=True),
    latent_size=32,
    hidden_dim=384
)
```

### 4.2 Nhóm thử nghiệm (Test Group)

```python
# Model B: Embedding Table thay thế CLIP
class TagEmbedding(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.tag_to_id = load_tag_vocabulary()  # Top 5000 tags
    
    def forward(self, tags: List[str]) -> Tensor:
        ids = [self.tag_to_id.get(t, 0) for t in tags]
        embeds = self.embedding(torch.tensor(ids))
        return embeds.mean(dim=0)  # Average pooling

model_b = DiT_Tiny(
    text_encoder=TagEmbedding(vocab_size=5000, embed_dim=768),
    latent_size=32,
    hidden_dim=384
)
```

### 4.3 Dataset

- **Nguồn:** Danbooru2021 subset
- **Số lượng:** 100,000 ảnh cho training, 10,000 cho evaluation
- **Preprocessing:** Resize về 256×256, extract top tags (max 20 tags/ảnh)

### 4.4 Metrics đo lường

| Metric | Công cụ | Mục tiêu |
|--------|---------|----------|
| **Preprocessing Time** | `time.perf_counter()` | Model B < 10% của Model A |
| **RAM Usage** | `psutil.Process().memory_info()` | Model B giảm 20-30% |
| **FID Score** | `pytorch-fid` | Model B ≈ Model A (±5%) |
| **Tag Accuracy** | Pretrained tagger (WD-Tagger) | Model B ≥ 90% của Model A |

---

## 5. Rủi ro và giới hạn

### 5.1 Rủi ro tiềm ẩn

| Rủi ro | Mức độ | Giải pháp dự phòng |
|--------|--------|-------------------|
| Embedding không học được semantic relationships | Trung bình | Thêm positional encoding hoặc learnable [SEP] tokens |
| OOV tags (ngoài top 5000) | Thấp | Fallback về [UNK] token hoặc nearest neighbor lookup |
| Overfitting do vocab nhỏ | Thấp | Regularization, dropout trên embedding |

### 5.2 Giới hạn của nghiên cứu

- **Không áp dụng cho text prompts tự nhiên:** Giả thuyết chỉ đúng với structured tags
- **Domain-specific:** Kết quả có thể không generalize sang photo-realistic images
- **Phụ thuộc vào tag quality:** Cần metadata chính xác từ Danbooru

---

## 6. Timeline dự kiến

```
Tuần 1: Data Preparation
├── [_] Thu thập và clean top 5000 tags
├── [_] Build vocabulary mapping
└── [_] Prepare dataset splits

Tuần 2: Implementation
├── [_] Implement TagEmbedding layer
├── [_] Integrate với DiT-Tiny
└── [_] Setup training pipeline

Tuần 3: Training & Evaluation
├── [_] Train cả 2 models (same compute budget)
├── [_] Benchmark speed/RAM
└── [_] Calculate FID & Tag Accuracy

Tuần 4: Analysis & Report
├── [_] Statistical significance tests
├── [_] Ablation studies
└── [_] Write final report
```

---

## 7. Kết quả kỳ vọng

### Nếu $H_1$ được chấp nhận:

- **Contribution:** Một phương pháp mới để tối ưu text conditioning cho domain-specific generation
- **Practical impact:** CPU users có thể chạy model nhanh hơn đáng kể
- **Follow-up research:** Hierarchical embeddings, Tag clustering, Multi-level conditioning

### Nếu $H_1$ bị bác bỏ:

- **Learning:** Xác nhận rằng CLIP's semantic understanding là cần thiết ngay cả với structured tags
- **Pivot:** Có thể thử hybrid approach (small CLIP + embedding table)

---

## 8. Câu hỏi mở cho phản biện

1. **Về thiết kế thí nghiệm:**
   - Liệu 5000 tags có đủ để cover phần lớn use cases?
   - Có nên dùng pretrained word embeddings (Word2Vec, FastText) thay vì random init?

2. **Về metrics:**
   - FID có phải là metric phù hợp nhất cho Anime?
   - Có cần Human evaluation không?

3. **Về tính khả thi:**
   - Làm sao xử lý tag combinations mà model chưa thấy trong training?
   - Weight của mỗi tag có nên bằng nhau không (average pooling)?

4. **Về scope:**
   - Có nên mở rộng thí nghiệm sang các domain khác (game assets, icons)?
   - Liệu kết quả có reproducible trên các CPU architectures khác nhau?

---

> **Ghi chú cho reviewer:** Vui lòng phản biện các điểm yếu trong thiết kế thí nghiệm, cơ sở lý luận, hoặc đề xuất cải tiến. Mục tiêu là làm cho giả thuyết này robust hơn trước khi tiến hành thực nghiệm.
