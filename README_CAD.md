# CAD System for Male Infertility 🔬

Hệ thống **Chẩn đoán Hỗ trợ Máy tính (CAD)** phát hiện và phân loại tinh trùng từ ảnh kính hiển vi quang học, phục vụ đánh giá chất lượng tinh dịch và hỗ trợ chẩn đoán vô sinh nam.

---

## Bối cảnh y học

### Kiểm tra chất lượng tinh trùng là gì?

Xét nghiệm tinh dịch đồ (semen analysis) là bước đầu tiên trong chẩn đoán vô sinh nam. Theo tiêu chuẩn WHO 2021, bác sĩ đánh giá tinh trùng qua 3 tiêu chí chính:

| Tiêu chí | Mô tả | Ngưỡng bình thường (WHO) |
|----------|--------|--------------------------|
| **Mật độ** | Số tinh trùng / mL | ≥ 16 triệu/mL |
| **Độ di động** | % tinh trùng di chuyển được | ≥ 42% tổng thể |
| **Hình thái** | % tinh trùng có hình dạng bình thường | ≥ 4% (Kruger) |

Ngoài 3 tiêu chí trên, **độ phân mảnh DNA (DNA Fragmentation Index – DFI)** ngày càng được coi trọng vì ảnh hưởng trực tiếp đến khả năng thụ tinh và phát triển phôi, kể cả khi các chỉ số thông thường còn bình thường.

### DFI và phương pháp SCD

Phương pháp **Sperm Chromatin Dispersion (SCD)** là kỹ thuật phổ biến để đo DFI:

1. Tinh trùng được xử lý hóa chất để biến tính DNA.
2. Quan sát dưới kính hiển vi: tinh trùng **có** DNA nguyên vẹn sẽ tạo ra **vầng sáng (halo)** xung quanh đầu; tinh trùng **bị phân mảnh DNA** không tạo halo hoặc halo rất nhỏ.
3. Kích thước halo là proxy đo mức độ tổn thương chromatin.

```
Halo lớn   → DNA ít phân mảnh → chất lượng tốt
Halo vừa   → DNA phân mảnh trung bình
Halo nhỏ   → DNA phân mảnh nhiều
Không halo → DNA phân mảnh nặng → nguy cơ vô sinh cao
```

### Khi nào chẩn đoán vô sinh nam?

Vô sinh nam được chẩn đoán khi cặp đôi không có thai sau ≥ 12 tháng quan hệ không bảo vệ, và kết quả tinh dịch đồ cho thấy ít nhất một bất thường:

- **Thiểu tinh (Oligospermia):** mật độ < 16 triệu/mL
- **Yếu tinh (Asthenospermia):** độ di động < 42%
- **Dị dạng tinh (Teratospermia):** hình thái bình thường < 4%
- **DFI cao:** > 15–25% tinh trùng bị phân mảnh DNA (ngưỡng tùy phòng lab)
- Kết hợp nhiều bất thường → **OAT syndrome**

---

## Về dự án này

Dự án xây dựng pipeline Deep Learning tự động hóa bước phân tích hình ảnh SCD, thay thế việc đếm tay của phôi thai học viên — vốn tốn thời gian và phụ thuộc kinh nghiệm người quan sát.

**Dataset:** [Expert-Annotated Optical Microscopy Images of Human Sperm](https://doi.org/10.6084/m9.figshare.30120811) (Saadat et al., Scientific Data 2025/2026), được gán nhãn độc lập bởi 5 phôi thai học viên, nhãn cuối dùng majority voting.

---

## Pipeline xử lý

```
Ảnh kính hiển vi thô (SCD)
        │
        ▼
  GD1: Preprocessing
   ├── Resize & chuẩn hóa kích thước
   ├── Chuyển không gian màu (RGB → Grayscale / LAB)
   ├── Khử nhiễu (Gaussian / Median filter)
   ├── Tăng tương phản (CLAHE)
   ├── Chuẩn hóa nền (background normalization)
   └── Data Augmentation (flip, rotate, crop…)
        │
        ▼
  GD2: Dataset Split  (70% train / 15% val / 15% test)
   ├── Task A: Binary Classification (Sperm / Non-sperm)
   └── Task B: Halo Classification (Large / Medium / Small / Without)
        │
        ▼
  GD3: Model Training
   ├── Task A: ResNet-50 (fine-tune, freeze backbone)
   └── Task B: EfficientNet-B0 (fine-tune, freeze backbone)
        │
        ▼
  GD4: Grad-CAM + Báo cáo
   ├── Grad-CAM: trực quan hóa vùng mà model chú ý
   └── So sánh với baseline (Majority Vote, Random)
```

---

## Cấu trúc project

```
CAD-System-Male-Infertility/
├── main.ipynb            ← Notebook chính (4 giai đoạn đầy đủ)
├── Metadata.csv          ← Nhãn và thông tin từng ảnh trong dataset
├── flowchart.svg         ← Sơ đồ pipeline
├── paper.pdf             ← Bài báo gốc (tiếng Anh)
├── paper-translation.pdf ← Bản dịch bài báo
└── README.txt            ← Mô tả dataset gốc
```

---

## Dataset

| Subset | Mục đích | Nhãn |
|--------|----------|------|
| **Binary Classification** | Phát hiện tinh trùng | `sperm` / `non-sperm` |
| **Halo Classification** | Đánh giá phân mảnh DNA | `large` / `medium` / `small` / `without halo` |
| **Raw Images** | Segmentation, clustering | Ảnh đầy đủ trường nhìn |

**Phân bố Task A (Binary):**
- Sperm: 320 ảnh
- Non-sperm: 330 ảnh
- Tổng: 650 ảnh → Train 454 / Val 98 / Test 98

---

## Mô hình & Kết quả

### Task A — Phát hiện tinh trùng (Binary Classification)

**Mô hình:** ResNet-50 (pretrained ImageNet, fine-tune lớp cuối)

| Metric | Giá trị |
|--------|---------|
| Accuracy | **84.69%** |
| Precision | 82.35% |
| Recall | 87.50% |
| F1-score | 84.85% |
| **AUC-ROC** | **94.08%** |

### Task B — Phân loại halo / đánh giá phân mảnh DNA (Multiclass)

**Mô hình:** EfficientNet-B0 (pretrained ImageNet, fine-tune lớp cuối)

| Metric | Giá trị |
|--------|---------|
| Accuracy | **55.74%** |
| Macro F1 | 56.77% |

**Per-class:**

| Lớp halo | Precision | Recall | Ý nghĩa lâm sàng |
|----------|-----------|--------|------------------|
| Large | 66.7% | 62.5% | DNA nguyên vẹn |
| Medium | 33.3% | 43.8% | Phân mảnh trung bình |
| Small | 50.0% | 42.9% | Phân mảnh nhiều |
| Without | 84.6% | 73.3% | Phân mảnh nặng |

> **Nhận xét:** Task A đạt AUC cao (0.94), phù hợp ứng dụng sàng lọc. Task B khó hơn do ranh giới giữa các lớp halo mờ nhạt và chủ quan — đây cũng là thách thức chung của bài toán DFI tự động.

---

## Cài đặt & Chạy

**Yêu cầu:** Python 3.8+, GPU khuyến nghị (Google Colab hoặc CUDA)

```bash
pip install torch torchvision opencv-python albumentations scikit-learn matplotlib pandas
```

**Chạy trên Google Colab:**

1. Upload `main.ipynb` lên Colab.
2. Mount Google Drive để lưu checkpoint model.
3. Tải dataset từ [Figshare DOI](https://doi.org/10.6084/m9.figshare.30120811) và giải nén vào `/content/extracted/Dataset/`.
4. Chạy tuần tự 4 giai đoạn: GD1 → GD2 → GD3 → GD4.

---

## Trích dẫn

```
Saadat H, Torkashvand H, Borna MR.
Expert-annotated optical microscopy images of human sperm
for detection and DNA fragmentation assessment.
Figshare, 2025. https://doi.org/10.6084/m9.figshare.30120811
```

---

## Giấy phép

Dataset gốc: **CC BY 4.0** — cho phép sử dụng, chia sẻ và chỉnh sửa với điều kiện ghi rõ nguồn.

---

> **Lưu ý:** Hệ thống này là công cụ hỗ trợ nghiên cứu, không thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa.
