"""Streamlit app for Vietnamese keyword extraction and API-based summarization."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from nlp_summary.evaluation.llm_benchmark import build_benchmark_row
from nlp_summary.evaluation.rouge import RougeEvaluator
from nlp_summary.keywords.registry import ExtractorRegistry
from nlp_summary.preprocessing.pipeline import PreprocessingPipeline
from nlp_summary.summarizer.llm_summarizer import (
    LLMSummaryResult,
    SemanticAnchoredLLMSummarizer,
)
from nlp_summary.ui.utils.file_parser import DocumentParser

MODEL_LABELS = {
    "qwen/qwen3.8-27b": "Qwen 3.8 27B — mô hình chính",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash — đối chứng tốc độ",
    "deepseek/deepseek-v3": "DeepSeek V3 — đối chứng mã nguồn mở",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna — đối chứng API thương mại",
}
DOCUMENT_TYPES = ["Bài báo", "Tiểu luận", "Văn bản hành chính"]
KEYWORD_LABELS = {
    "tfidf": "TF-IDF + từ loại (baseline)",
    "keybert": "KeyBERT + Vietnamese bi-encoder (advanced)",
}


@st.cache_resource
def get_preprocessor() -> PreprocessingPipeline:
    return PreprocessingPipeline(language="vi")


@st.cache_resource
def get_keyword_extractor(algorithm: str):
    # Caching is important for KeyBERT because the embedding model is loaded lazily.
    return ExtractorRegistry.get_extractor(algorithm)


def extract_uploaded_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    return DocumentParser.extract_text(uploaded_file.name, uploaded_file.getvalue())


def keyword_table(keywords) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Từ khóa/cụm từ": keyword.keyword.replace("_", " "),
                "Điểm chuẩn hóa": keyword.normalized_score,
                "Điểm gốc": keyword.raw_score,
                "Số từ": keyword.ngram_length,
                "Thuật toán": keyword.algorithm,
            }
            for keyword in keywords
        ]
    )


def show_llm_result(result: LLMSummaryResult, reference_summary: str = "") -> None:
    if result.is_fallback:
        st.warning(
            "API không tạo được bản tóm tắt. Kết quả dưới đây là bản trích xuất dự phòng, "
            "không được tính là đầu ra LLM."
        )
        if result.fallback_reason:
            st.caption(result.fallback_reason)
    else:
        st.success(f"Đã sinh tóm tắt bằng `{result.model_used}`.")

    st.write(result.summary or "Không có nội dung trả về.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Độ trễ API", f"{result.latency_ms:,.0f} ms")
    metric_columns[1].metric("Độ phủ semantic anchors", f"{result.anchor_coverage:.1%}")
    metric_columns[2].metric("Trung thực số liệu", f"{result.number_faithfulness:.1%}")
    metric_columns[3].metric("Tỷ lệ độ dài", f"{result.compression_ratio:.1%}")

    if result.unsupported_numbers:
        st.error(
            "Các số xuất hiện trong tóm tắt nhưng không tìm thấy trong nguồn: "
            + ", ".join(result.unsupported_numbers)
        )

    if reference_summary.strip() and result.summary.strip() and not result.is_fallback:
        rouge = RougeEvaluator().evaluate(result.summary, reference_summary)
        st.caption(
            "ROUGE với bản tham chiếu — "
            f"R-1 F1: {rouge.rouge_1.f1_score:.3f}; "
            f"R-2 F1: {rouge.rouge_2.f1_score:.3f}; "
            f"R-L F1: {rouge.rouge_l.f1_score:.3f}."
        )


st.set_page_config(
    page_title="Tóm tắt văn bản tiếng Việt bằng từ khóa và LLM",
    page_icon=":material/summarize:",
    layout="wide",
)

st.title("Tóm tắt văn bản tiếng Việt bằng semantic anchors")
st.caption(
    "Tiền xử lý tiếng Việt → TF-IDF/KeyBERT → từ khóa cốt lõi → LLM API → kiểm tra kết quả"
)

with st.sidebar:
    st.header("Cấu hình API")
    api_key = st.text_input(
        "Haimaker API key",
        type="password",
        key="session_api_key",
        placeholder="Dán API key để bắt đầu demo",
        help="Khóa chỉ dùng trong phiên Streamlit hiện tại; app không ghi key xuống tệp hoặc kết quả.",
    )
    api_endpoint = st.text_input(
        "OpenAI-compatible endpoint",
        value=SemanticAnchoredLLMSummarizer.DEFAULT_ENDPOINT,
        key="api_endpoint",
    )
    if api_key:
        st.success("Đã nhận API key cho phiên Streamlit hiện tại.")
    else:
        st.info("Nhập API key để sinh tóm tắt hoặc chạy đối chứng LLM.")
    st.caption("App chỉ gọi API sau khi bạn bấm nút sinh tóm tắt hoặc chạy đối chứng.")

view = st.segmented_control(
    "Chế độ làm việc",
    ["Tóm tắt một văn bản", "Đối chứng LLM", "Phương pháp"],
    default="Tóm tắt một văn bản",
    key="active_view",
)

if view == "Tóm tắt một văn bản":
    st.subheader("Dữ liệu và cấu hình thí nghiệm")
    input_mode = st.segmented_control(
        "Nguồn văn bản",
        ["Nhập trực tiếp", "Tải tệp TXT/PDF"],
        default="Nhập trực tiếp",
        key="input_mode",
    )

    with st.form("summarization_form"):
        if input_mode == "Nhập trực tiếp":
            direct_text = st.text_area(
                "Văn bản tiếng Việt",
                height=260,
                placeholder="Dán bài báo, tiểu luận hoặc văn bản hành chính tại đây...",
            )
            uploaded_file = None
        else:
            direct_text = ""
            uploaded_file = st.file_uploader(
                "Tệp đầu vào",
                type=["txt", "pdf"],
                help="PDF cần có lớp văn bản; app không thực hiện OCR trong chế độ này.",
            )

        config_columns = st.columns(3)
        document_type = config_columns[0].selectbox("Loại văn bản", DOCUMENT_TYPES)
        keyword_algorithm = config_columns[1].selectbox(
            "Trích xuất từ khóa",
            list(KEYWORD_LABELS),
            format_func=KEYWORD_LABELS.get,
        )
        top_k = config_columns[2].slider("Số semantic anchors", 3, 15, 7)

        output_columns = st.columns(3)
        selected_model = output_columns[0].selectbox(
            "LLM API",
            list(MODEL_LABELS),
            format_func=MODEL_LABELS.get,
        )
        max_sentences = output_columns[1].slider("Số câu tối đa", 2, 8, 4)
        max_words = output_columns[2].slider("Số từ tối đa", 60, 300, 180, step=20)

        reference_summary = st.text_area(
            "Bản tóm tắt tham chiếu để tính ROUGE (không bắt buộc)",
            height=100,
        )
        submitted = st.form_submit_button(
            "Trích xuất từ khóa và sinh tóm tắt",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if submitted:
        raw_text = direct_text.strip() or extract_uploaded_text(uploaded_file).strip()
        if not raw_text:
            st.error("Không đọc được nội dung văn bản. Hãy kiểm tra dữ liệu đầu vào.")
        elif len(raw_text) < 50:
            st.error("Văn bản quá ngắn; cần ít nhất 50 ký tự để tóm tắt có ý nghĩa.")
        else:
            with st.status("Đang xử lý văn bản...", expanded=True) as status:
                preprocessing_started = time.perf_counter()
                document = get_preprocessor().process(raw_text)
                preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
                st.write(f"Tiền xử lý: {preprocessing_ms:,.0f} ms")

                keyword_started = time.perf_counter()
                extractor = get_keyword_extractor(keyword_algorithm)
                keywords = extractor.extract_keywords(document, top_k=top_k)
                keyword_ms = (time.perf_counter() - keyword_started) * 1000
                st.write(f"Trích xuất {len(keywords)} từ khóa: {keyword_ms:,.0f} ms")

                if not keywords:
                    status.update(
                        label="Không trích xuất được từ khóa",
                        state="error",
                    )
                    st.error(
                        "Thuật toán không trả về semantic anchors; app không gọi LLM "
                        "để tránh một thí nghiệm sai cấu hình."
                    )
                    st.stop()

                actual_keyword_algorithms = sorted(
                    {keyword.algorithm for keyword in keywords}
                )
                if keyword_algorithm not in actual_keyword_algorithms:
                    st.warning(
                        f"Đã chọn `{keyword_algorithm}` nhưng backend thực tế trả về "
                        f"`{', '.join(actual_keyword_algorithms)}`. Kết quả này không được "
                        f"báo cáo như một lần chạy {keyword_algorithm}."
                    )

                summarizer = SemanticAnchoredLLMSummarizer(
                    api_key=api_key,
                    api_endpoint=api_endpoint,
                    default_model=selected_model,
                )
                llm_result = summarizer.summarize(
                    document,
                    keywords,
                    model=selected_model,
                    top_anchors_count=top_k,
                    max_sentences=max_sentences,
                    max_words=max_words,
                    document_type=document_type,
                    fallback_on_error=True,
                )
                status.update(label="Đã hoàn tất pipeline", state="complete")

            st.session_state["experiment"] = {
                "raw_text": raw_text,
                "document": document,
                "document_type": document_type,
                "keyword_algorithm": keyword_algorithm,
                "actual_keyword_algorithms": actual_keyword_algorithms,
                "keywords": keywords,
                "top_k": top_k,
                "max_sentences": max_sentences,
                "max_words": max_words,
                "reference_summary": reference_summary,
                "llm_result": llm_result,
            }
            st.session_state.pop("benchmark_results", None)

    experiment = st.session_state.get("experiment")
    if experiment:
        keyword_column, summary_column = st.columns([0.42, 0.58])
        with keyword_column:
            st.subheader("Từ khóa cốt lõi")
            st.dataframe(
                keyword_table(experiment["keywords"]),
                hide_index=True,
                width="stretch",
                key="primary_keywords",
            )
        with summary_column:
            st.subheader("Bản tóm tắt")
            show_llm_result(
                experiment["llm_result"], experiment["reference_summary"]
            )

        download_text = (
            f"MÔ HÌNH: {experiment['llm_result'].model_used}\n"
            f"THUẬT TOÁN TỪ KHÓA: {experiment['keyword_algorithm']}\n"
            "TỪ KHÓA: "
            + ", ".join(
                keyword.keyword.replace("_", " ")
                for keyword in experiment["keywords"]
            )
            + f"\n\nBẢN TÓM TẮT:\n{experiment['llm_result'].summary}"
        )
        st.download_button(
            "Tải kết quả TXT",
            data=download_text,
            file_name="ket_qua_tom_tat.txt",
            mime="text/plain",
            icon=":material/download:",
        )

elif view == "Đối chứng LLM":
    st.subheader("Đối chứng các LLM mới trên cùng điều kiện")
    experiment = st.session_state.get("experiment")
    if not experiment:
        st.info("Hãy chạy một văn bản ở chế độ “Tóm tắt một văn bản” trước.")
    else:
        actual_algorithms = experiment.get(
            "actual_keyword_algorithms", [experiment["keyword_algorithm"]]
        )
        st.write(
            f"Đầu vào cố định: **{experiment['document_type']}** · "
            f"**{len(experiment['raw_text'].split()):,} từ** · "
            f"**{KEYWORD_LABELS[experiment['keyword_algorithm']]}** · "
            f"backend **{', '.join(actual_algorithms)}** · "
            f"**{len(experiment['keywords'])} anchors**"
        )
        st.caption(
            "Mọi model nhận cùng văn bản, prompt, semantic anchors, nhiệt độ 0.2 và ràng buộc độ dài. "
            "Fallback bị tắt để không làm sai lệch kết quả đối chứng."
        )

        with st.form("benchmark_form"):
            comparison_models = st.multiselect(
                "Các model cần đối chứng",
                list(MODEL_LABELS),
                default=list(MODEL_LABELS)[:3],
                format_func=MODEL_LABELS.get,
            )
            benchmark_reference = st.text_area(
                "Bản tóm tắt tham chiếu",
                value=experiment["reference_summary"],
                height=100,
                help="Có bản tham chiếu thì bảng sẽ tính thêm ROUGE-1/2/L.",
            )
            run_benchmark = st.form_submit_button(
                f"Gọi API và đối chứng ({len(comparison_models)} lượt)",
                type="primary",
                icon=":material/compare_arrows:",
            )

        if run_benchmark:
            if not comparison_models:
                st.error("Hãy chọn ít nhất một model.")
            elif not api_key:
                st.error("Chưa có API key; không thể chạy đối chứng.")
            else:
                results = []
                progress = st.progress(0, text="Bắt đầu gọi API...")
                for index, model_name in enumerate(comparison_models, start=1):
                    progress.progress(
                        (index - 1) / len(comparison_models),
                        text=f"Đang chạy {MODEL_LABELS[model_name]}...",
                    )
                    summarizer = SemanticAnchoredLLMSummarizer(
                        api_key=api_key,
                        api_endpoint=api_endpoint,
                        default_model=model_name,
                    )
                    result = summarizer.summarize(
                        experiment["document"],
                        experiment["keywords"],
                        model=model_name,
                        top_anchors_count=experiment["top_k"],
                        max_sentences=experiment["max_sentences"],
                        max_words=experiment["max_words"],
                        document_type=experiment["document_type"],
                        fallback_on_error=False,
                    )
                    results.append(result)
                progress.progress(1.0, text="Đã hoàn tất đối chứng API.")
                st.session_state["benchmark_results"] = {
                    "results": results,
                    "reference_summary": benchmark_reference,
                }

        benchmark = st.session_state.get("benchmark_results")
        if benchmark:
            rows = [
                build_benchmark_row(result, benchmark["reference_summary"]).model_dump()
                for result in benchmark["results"]
            ]
            benchmark_df = pd.DataFrame(rows).rename(
                columns={
                    "model": "Model",
                    "status": "Trạng thái",
                    "latency_ms": "Độ trễ (ms)",
                    "word_count": "Số từ",
                    "compression_ratio": "Tỷ lệ độ dài",
                    "anchor_coverage": "Độ phủ anchors",
                    "number_faithfulness": "Trung thực số liệu",
                    "rouge_1_f1": "ROUGE-1 F1",
                    "rouge_2_f1": "ROUGE-2 F1",
                    "rouge_l_f1": "ROUGE-L F1",
                    "error": "Lỗi",
                }
            )
            st.dataframe(benchmark_df, hide_index=True, width="stretch", key="llm_benchmark")
            st.download_button(
                "Tải bảng đối chứng CSV",
                data=benchmark_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="doi_chung_llm.csv",
                mime="text/csv",
                icon=":material/download:",
            )

            for result in benchmark["results"]:
                with st.expander(result.model_used):
                    if result.is_fallback:
                        st.error(result.fallback_reason or "API không trả về kết quả.")
                    else:
                        st.write(result.summary)

else:
    st.subheader("Thiết kế hệ thống và nguyên tắc đối chứng")
    st.markdown(
        """
1. **Đầu vào:** một bài báo, tiểu luận hoặc văn bản hành chính tiếng Việt từ nhập liệu, TXT hoặc PDF.
2. **Tiền xử lý:** chuẩn hóa, tách câu, tách từ và gán nhãn từ loại bằng Underthesea; loại stop words tiếng Việt.
3. **Baseline:** TF-IDF chỉ giữ danh từ, động từ và tính từ.
4. **Advanced:** KeyBERT sử dụng `bkai-foundation-models/vietnamese-bi-encoder`.
5. **Sinh tóm tắt:** danh sách từ khóa trở thành semantic anchors trong prompt gửi đến LLM API.
6. **Đối chứng:** các model nhận cùng đầu vào và tham số; app ghi nhận kết quả thật, không dùng số liệu mô phỏng.
        """
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["Độ phủ semantic anchors", "Tỷ lệ anchors xuất hiện trong tóm tắt", "Không cần bản tham chiếu"],
                ["Trung thực số liệu", "Tỷ lệ số trong tóm tắt có mặt trong nguồn", "Không cần bản tham chiếu"],
                ["Độ trễ", "Thời gian từ lúc gửi đến khi nhận API response", "Đo trực tiếp"],
                ["ROUGE-1/2/L", "Mức trùng khớp với bản tóm tắt chuẩn", "Cần bản tham chiếu"],
                ["Tỷ lệ độ dài", "Số từ tóm tắt / số từ văn bản nguồn", "Không cần bản tham chiếu"],
            ],
            columns=["Chỉ số", "Ý nghĩa", "Điều kiện"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "Các chỉ số tự động không chứng minh tuyệt đối rằng bản tóm tắt không ảo giác. "
        "Kết quả nghiên cứu vẫn cần đánh giá thủ công trên một mẫu được mô tả rõ."
    )
