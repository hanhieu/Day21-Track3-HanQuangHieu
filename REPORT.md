# Lab 21 - Evaluation Report

**Tên:** Hàn Quang Hiếu  
**Mã học viên:** 2A202600056  
**Ngày nộp:** 2026-05-07  
**Submission option:** B - GitHub + Hugging Face Hub

## 1. Setup

- **Base model:** `unsloth/Qwen2.5-3B-bnb-4bit`
- **Bài lab sử dụng notebook:** `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`
- **Dataset:** `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Số lượng mẫu:** 200 samples
- **Train / Eval split:** 180 train / 20 eval
- **Format dữ liệu:** Alpaca-style với auto-detect các cột `instruction_vi`, `input_vi`, `output_vi`
- **Token length analysis:** `p50 = 227`, `p95 = 562`, `p99 = 704`
- **max_seq_length:** `1024` (round up từ p95 và bị cap bởi cấu hình T4)
- **GPU profile:** Google Colab T4 16 GB theo đúng profile của notebook và thư mục kết quả `google colab/lab21_lora_t4/`
- **Precision / quantization:** QLoRA 4-bit NF4, `dtype=None` để auto BF16/FP16, gradient checkpointing `unsloth`
- **LoRA target modules:** `["q_proj", "v_proj"]`
- **Hyperparameters chính:** batch size = 1, gradient accumulation = 8, 3 epochs, learning rate = `2e-4`, cosine scheduler, `optim="adamw_8bit"`, `packing=False`
- **Hugging Face adapter link:** https://huggingface.co/hanhieu/qwen2.5-3b-vi-lab21-r16

### Local implementation on RTX 3060 12 GB

Ngoài việc chạy và thu thập kết quả từ môi trường Google Colab T4, tôi cũng chủ động triển khai thêm code để thử chạy bài lab trên máy local với **NVIDIA GeForce RTX 3060 12 GB**. Phần này được thực hiện nhằm kiểm tra khả năng tái lập pipeline QLoRA trên GPU cá nhân, đồng thời hiểu rõ hơn các rào cản khi chuyển notebook từ Colab sang Windows native.

Các phần tôi đã implement thêm gồm:

- viết script local tại `scripts/run_lab21_local.py` để tái hiện logic chính của notebook;
- bổ sung launcher PowerShell cho Unsloth local;
- tạo môi trường ảo riêng để cô lập dependency và tránh xung đột với Python hệ thống;
- kiểm tra trực tiếp GPU local: PyTorch nhận đúng `RTX 3060`, CUDA khả dụng và `bf16_supported = True`.

Tuy nhiên, trong quá trình chạy local với Unsloth trên Windows native, tôi gặp một số lỗi quan trọng:

1. **Xung đột dependency giữa notebook cũ và Unsloth mới**
   - Notebook gốc pin `trl >= 0.12, < 0.16`, trong khi các phiên bản Unsloth hiện tại đã yêu cầu stack mới hơn (`trl >= 0.18.2`).
   - Điều này buộc tôi phải điều chỉnh lại dependency nếu muốn chạy local với package mới.

2. **Lỗi quyền ghi / môi trường khi tạo venv trong repo**
   - Lúc đầu việc tạo `venv` trong thư mục project bị lỗi ở bước `ensurepip` do vấn đề quyền ghi trên thư mục temp của Windows.
   - Sau đó tôi phải đổi hướng sang dùng thư mục temp ngắn hơn và môi trường riêng ở ngoài repo.

3. **Lỗi Windows long-path khi cài PyTorch / xformers trong venv nằm trong repo**
   - Việc cài stack CUDA vào venv dạng `.venv` trong project gặp lỗi `OSError` do đường dẫn file quá dài trên Windows.
   - Cách khắc phục thực tế là chuyển venv sang đường dẫn ngắn hơn như `C:\tmp\u21venv`.

4. **Lỗi encoding console trên Windows**
   - Khi Unsloth in log có emoji, console Windows dùng `cp1252` gây `UnicodeEncodeError`.
   - Tôi phải cấu hình lại output UTF-8 để tránh lỗi này.

5. **Lỗi quan trọng nhất: thiếu MSVC compiler (`cl.exe`)**
   - Mặc dù GPU local được nhận đúng, Unsloth vẫn không train được trên Windows native vì Triton cần compiler C/C++ của Microsoft để build kernel.
   - Khi chạy local, quá trình train dừng ở lỗi:
     - `RuntimeError: Failed to find C compiler. Please specify via CC environment variable.`
     - launcher sau đó xác nhận rõ: `MSVC cl.exe not found.`

6. **Kết luận về local run**
   - Về phần cứng, **RTX 3060 12 GB hoàn toàn đủ để chạy QLoRA 3B**.
   - Nhưng về phần mềm, **Unsloth trên Windows native không chỉ cần GPU và PyTorch**, mà còn phụ thuộc vào toolchain C++/Triton.
   - Vì vậy, bottleneck của local run trong trường hợp này không nằm ở VRAM mà nằm ở môi trường hệ điều hành và compiler.

### Training cost ước tính

File `rank_experiment_summary.csv` cho tổng thời gian train của 3 rank là:

- `r=8`: 3.74 phút
- `r=16`: 4.15 phút
- `r=64`: 3.72 phút
- **Tổng:** 11.61 phút

Notebook dùng rate mặc định của T4 là `$0.35 / giờ`, nên:

- **Estimated training cost:** khoảng **$0.07**

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------------|-----------|-----------|------------|
| 8 | 16 | 1,843,200 | 3.74 min | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.15 min | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 3.72 min | 8.00 GB | 1.4768 | 4.3790 |

### Nhận xét nhanh từ bảng kết quả

- Về **quality**, rank càng cao thì perplexity càng thấp: `r=64` tốt nhất, tiếp theo là `r=16`, rồi `r=8`.
- Về **VRAM**, `r=16` lại là cấu hình tiết kiệm VRAM nhất trong kết quả đã lưu, chỉ khoảng `6.62 GB`, thấp hơn `r=8` và thấp hơn đáng kể so với `r=64`.
- Về **training time**, chênh lệch giữa 3 rank là không lớn trên tập 200 mẫu. `r=16` chậm nhất nhưng chỉ nhiều hơn khoảng 0.4 phút so với `r=8`.
- Về **trainable parameters**, `r=64` lớn gấp 4 lần `r=16` và gấp 8 lần `r=8`, nhưng đổi lại mức cải thiện perplexity không tăng tương ứng với mức tăng tham số.

### Insight chính về trade-off

- **`r=8`** là cấu hình nhẹ nhất về số trainable params, phù hợp khi cần tiết kiệm adapter size và muốn thử nghiệm nhanh.
- **`r=16`** cho một điểm cân bằng tốt giữa chất lượng và tài nguyên: perplexity cải thiện rõ so với `r=8`, trong khi VRAM lại thấp nhất trong 3 cấu hình.
- **`r=64`** cho chất lượng tốt nhất trên tập eval này, nhưng cái giá phải trả là adapter lớn hơn nhiều và peak VRAM cao nhất.

## 3. Loss Curve Analysis

Notebook T4 profile được thiết kế với `eval_strategy = "no"` để tránh OOM trên T4 trong lúc đang train. Vì vậy:

- Trong quá trình train chỉ có **train loss curve**, không có eval loss curve theo từng step.
- Bộ artifacts hiện tại không kèm riêng file `loss_curve.png`, nhưng logic trong notebook cho thấy tác giả có chủ đích tắt mid-train eval để ưu tiên tính ổn định trên GPU nhỏ.

### Phân tích

- Vì không có eval loss theo từng bước, **không thể kết luận chắc chắn overfitting hay không** chỉ từ artifacts hiện có.
- Tuy nhiên, nhìn vào kết quả cuối cùng:
  - cả 3 rank đều cho perplexity khá thấp và ổn định quanh `4.38 - 4.75`,
  - không có dấu hiệu rank lớn hơn làm chất lượng trên eval xấu đi.
- Điều này gợi ý rằng trong bài lab này, mô hình vẫn đang học được tín hiệu hữu ích từ dataset 200 mẫu thay vì rơi vào tình trạng suy giảm ngay trên tập eval.

### Kết luận cho phần loss

- **Chưa có bằng chứng rõ ràng về overfitting** trong artifacts đã lưu.
- Tuy vậy, vì dataset khá nhỏ và không có eval-during-training, vẫn nên xem đây là một rủi ro tiềm ẩn nếu tăng epoch hoặc tăng rank hơn nữa.

## 4. Qualitative Comparison

Phần qualitative trong artifacts hiện tại lưu so sánh **base model vs fine-tuned model `r=16`** trên 5 prompt.

### Example 1

**Prompt:** Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base:** Trả lời đúng hướng, giải thích machine learning là một nhánh của AI và học từ dữ liệu, nhưng câu văn còn dài và hơi lan man.

**Fine-tuned (r=16):** Vẫn đúng ý chính, trình bày gọn hơn, nhấn mạnh rõ mối quan hệ giữa AI, thuật toán học máy và khả năng dự đoán từ dữ liệu.

**Nhận xét:** Fine-tuned output mạch lạc hơn và có cảm giác “instruction-following” rõ hơn.

### Example 2

**Prompt:** Viết đoạn code Python tính số Fibonacci thứ n.

**Base:** Đưa ra lời giải bằng Python nhưng thiên về giải thích chung và phần code bị cắt ngắn trong artifact.

**Fine-tuned (r=16):** Trả lời trực tiếp hơn, có kiểm tra input âm bằng `ValueError`, rồi dùng cách lặp `a, b = 0, 1`.

**Nhận xét:** Fine-tuned model có format trả lời sát yêu cầu lập trình hơn, thực dụng hơn và rõ ý hơn base.

### Example 3

**Prompt:** Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base:** Nêu được một số ý như thân thiện với người dùng, bố cục, màu sắc, font chữ, nhưng văn bản khá dài và chưa thật sự “liệt kê” gọn.

**Fine-tuned (r=16):** Trả lời dưới dạng danh sách ngắn hơn với các ý như chuyển đổi, thích ứng, đơn giản, tương thích.

**Nhận xét:** Fine-tuned model tuân theo định dạng liệt kê tốt hơn, nhưng chất lượng nội dung chưa chắc tốt hơn hoàn toàn vì một số nguyên tắc còn diễn đạt hơi chung chung.

### Example 4

**Prompt:** Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base:** Có cố gắng phân biệt hai khái niệm nhưng giải thích còn mơ hồ và dễ gây hiểu nhầm.

**Fine-tuned (r=16):** Vẫn trả lời trôi chảy hơn, nhưng phần định nghĩa “LoRA” và “QLoRA” trong artifact chưa thật chính xác về mặt kỹ thuật.

**Nhận xét:** Đây là ví dụ cho thấy fine-tuning cải thiện format và độ trôi chảy, nhưng **không tự động sửa được knowledge gap kỹ thuật**, đúng với bài học lý thuyết của Day 21.

### Example 5

**Prompt:** Phân biệt prompt engineering, RAG, và fine-tuning.

**Base:** Trả lời đúng chủ đề, phân biệt ba kỹ thuật nhưng diễn đạt dài và hơi lặp.

**Fine-tuned (r=16):** Output có cấu trúc tốt hơn, vào đúng ý hơn ngay từ đầu, mô tả ba khái niệm theo dạng phân biệt tương đối rõ.

**Nhận xét:** Fine-tuned model cải thiện tính trực diện và format theo instruction, dù chiều sâu tri thức vẫn chưa vượt trội rõ rệt.

### Tổng hợp qualitative

- Fine-tuned `r=16` nhìn chung **trả lời gọn hơn, đúng format hơn, và bám instruction tốt hơn** base model.
- Lợi ích nổi bật nhất của fine-tuning trong bài này là:
  - cải thiện độ nhất quán về style trả lời,
  - tăng khả năng “vào thẳng yêu cầu”,
  - trình bày output có cấu trúc hơn.
- Tuy nhiên, với các prompt mang tính kiến thức kỹ thuật sâu như LoRA vs QLoRA, fine-tuned model **vẫn có thể sai về facts**, cho thấy fine-tuning không thay thế RAG hay kiến thức nền của base model.

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả của bài lab này, tôi đánh giá **`r=16` là cấu hình cho ROI tốt nhất**. Lý do là `r=16` cải thiện perplexity rõ rệt so với `r=8` (`4.5544` so với `4.7479`) nhưng không làm tăng chi phí quá nhiều, thậm chí peak VRAM trong run này còn thấp nhất (`6.62 GB`). Trong khi đó, `r=64` đúng là cho kết quả tốt nhất về perplexity (`4.3790`), nhưng số trainable parameters tăng lên rất mạnh, từ `3.69M` của `r=16` lên `14.75M`. Mức cải thiện chất lượng là có, nhưng không tỷ lệ thuận với chi phí tham số và bộ nhớ.

Điểm đáng chú ý là hiện tượng **diminishing returns** đã bắt đầu xuất hiện. Từ `r=8` lên `r=16`, model có thêm một bước cải thiện tương đối rõ cả về perplexity lẫn độ ổn định qualitative. Nhưng từ `r=16` lên `r=64`, chất lượng chỉ tăng thêm một mức vừa phải, trong khi adapter lớn hơn nhiều và VRAM cũng cao hơn. Điều này cho thấy trên một dataset nhỏ khoảng 200 mẫu như bài lab, việc đẩy rank quá cao chưa chắc là lựa chọn tối ưu.

Nếu deploy trong production cho một use case tương tự, tôi sẽ ưu tiên **`r=16`**. Nó là điểm cân bằng hợp lý giữa chất lượng, độ gọn của adapter, khả năng chạy trên GPU nhỏ và chi phí vận hành. Tôi chỉ chọn `r=64` nếu có bằng chứng rõ rằng workload thực tế thực sự cần độ chính xác cao hơn và hạ tầng có đủ dư địa về VRAM và storage. Còn `r=8` phù hợp cho các trường hợp thử nghiệm nhanh, benchmark nhanh, hoặc khi cần giữ adapter cực nhẹ.

## 6. What I Learned

- Fine-tuning với LoRA/QLoRA chủ yếu giúp model học **style, format và hành vi trả lời**, chứ không tự động sửa được các thiếu hụt kiến thức nền.
- Rank lớn hơn không phải lúc nào cũng đáng giá; với dataset nhỏ, `r=16` thường là điểm cân bằng thực tế hơn `r=64`.
- Trên GPU nhỏ như T4, các quyết định về `max_seq_length`, gradient checkpointing, `packing=False`, và cách evaluate an toàn quan trọng không kém bản thân model.
- Khi chuyển notebook từ Colab sang máy local, phần khó nhất không hẳn là code train mà là **dependency management** và **tương thích hệ điều hành**.
- Tôi học được rằng trên Windows native, để chạy Unsloth ổn định cần quan tâm thêm đến **MSVC Build Tools, Triton, đường dẫn file, encoding console và version pinning**; GPU đủ mạnh thôi vẫn chưa đủ.
- Việc tôi tự viết thêm script local và debug các lỗi môi trường giúp tôi hiểu rõ hơn thế nào là một pipeline fine-tuning “production-like”: phải nghĩ tới khả năng tái lập, cô lập môi trường, và xử lý failure modes ngoài mô hình.

## Appendix

### Files/artifacts đã dùng để viết report

- `README.md`
- `Lab21_Rubric_and_Format.md`
- `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`
- `google colab/lab21_lora_t4/rank_experiment_summary.csv`
- `google colab/lab21_lora_t4/qualitative_comparison.csv`
- `google colab/lab21_lora_t4/r16/adapter_config.json`

### Hugging Face

- Model repo: https://huggingface.co/hanhieu/qwen2.5-3b-vi-lab21-r16
- README link provided: https://huggingface.co/hanhieu/qwen2.5-3b-vi-lab21-r16/blob/main/README.md
