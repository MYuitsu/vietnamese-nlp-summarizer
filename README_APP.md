# Vietnamese keyword-anchored LLM summarization

Web application xử lý một văn bản tiếng Việt để trích xuất từ khóa và sinh bản tóm tắt bằng LLM API.

## Chức năng theo yêu cầu

- Nhập trực tiếp hoặc đọc tệp TXT/PDF.
- Tách từ và gán nhãn từ loại bằng Underthesea.
- Loại stop words tiếng Việt tổng quát và hành chính.
- Baseline: TF-IDF kết hợp lọc danh từ, động từ và tính từ.
- Advanced: KeyBERT với `bkai-foundation-models/vietnamese-bi-encoder`.
- Dùng từ khóa làm semantic anchors trong prompt tóm tắt.
- Gọi LLM qua OpenAI-compatible API.
- Đối chứng nhiều model với cùng văn bản, prompt, anchors và tham số.
- Tính latency, độ phủ anchors, trung thực số liệu, tỷ lệ độ dài và ROUGE khi có bản tham chiếu.

## Cài đặt và chạy

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

Mở `http://localhost:8501`, sau đó dán Haimaker API key vào ô mật khẩu ở thanh bên trái. Key chỉ được giữ trong bộ nhớ của phiên Streamlit hiện tại và không được ghi xuống tệp hoặc kết quả tải xuống.

## Đưa app lên web bằng Streamlit Community Cloud

1. Tạo một repository GitHub và đưa mã nguồn này lên repository.
2. Truy cập `https://share.streamlit.io` và đăng nhập bằng GitHub.
3. Chọn **Create app** → **Yup, I have an app**.
4. Chọn repository, branch `main` và main file `streamlit_app.py`.
5. Trong **Advanced settings**, chọn Python 3.11. Không cần cấu hình Secrets.
6. Chọn **Deploy** và chờ cài đặt các gói trong `requirements.txt`.
7. Mở URL dạng `https://...streamlit.app`, nhập Haimaker API key trên giao diện rồi demo.

Mỗi người mở app có phiên Streamlit riêng và tự nhập key. Không đưa API key vào GitHub, URL, file TXT/CSV xuất ra hoặc ảnh chụp màn hình khi demo.

Lần đầu chọn KeyBERT, Community Cloud cần tải `bkai-foundation-models/vietnamese-bi-encoder`, nên sẽ chậm hơn TF-IDF. Nên mở app và chạy thử KeyBERT trước buổi trình bày để cache model.

## Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Bộ test gồm kiểm thử tiền xử lý, TF-IDF/POS, KeyBERT, mock LLM API, chỉ số đối chứng và smoke test Streamlit.

## Nguyên tắc đối chứng

Mỗi model nhận cùng đầu vào, semantic anchors, prompt, nhiệt độ và giới hạn độ dài. Khi API lỗi, chế độ đối chứng không dùng bản TextRank/MMR dự phòng để tính ROUGE. Các chỉ số tự động chỉ hỗ trợ phân tích; đánh giá nghiên cứu vẫn cần một mẫu được chuyên gia hoặc người chấm xem xét thủ công.
