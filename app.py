import logging
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

logging.getLogger(
    "langchain_milvus.vectorstores.milvus"
).setLevel(logging.ERROR)

from core.engine import AdaptiveRAG  # noqa: E402


def main():
    print("Dang khoi tao He thong Medical Adaptive RAG...")
    try:
        rag_system = AdaptiveRAG()
    except Exception as e:
        print(f"Loi khoi tao he thong (Kiem tra API keys): {e}")
        rag_system = None

    def respond(message, history, user_context, use_web):
        if not rag_system:
            return (
                "Khoi tao he thong that bai. Vui long kiem tra "
                "API keys trong file .env va khoi dong lai "
                "ung dung."
            )

        try:
            res = rag_system.answer(
                query=message,
                user_context=(
                    user_context.strip() if user_context
                    else None
                ),
                use_web=use_web,
            )

            answer_text = res["answer"]
            category = res["category"]
            sources = res["sources"]

            formatted_response = (
                f"**[Chien luoc {category} duoc kich hoat]**"
                f"\n\n{answer_text}\n\n---\n"
                "**Nguon tham khao:**\n"
            )
            if sources:
                formatted_sources = "\n".join(
                    f"- `{src}`" for src in set(sources)
                )
                formatted_response += formatted_sources
            else:
                formatted_response += (
                    "- Khong co nguon tai lieu noi bo nao "
                    "duoc su dung."
                )

            return formatted_response

        except Exception as e:
            return f"Loi khi xu ly cau hoi: {str(e)}"

    with gr.Blocks(title="Tro ly Y te RAG") as demo:
        gr.Markdown(
            "# Tro ly Y te Thong minh (Adaptive RAG)"
        )
        gr.Markdown(
            "Tro ly lam sang thong minh co kha nang tu dong "
            "lua chon chien luoc truy xuat phu hop "
            "(Thuc te, Phan tich, Quan diem, Theo ngu canh) "
            "cho cac cau hoi y te, duoc toi uu hoa bang "
            "phan tich tai lieu va xep hang lai da ngon ngu."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg_input = gr.Textbox(
                    placeholder=(
                        "Nhap cau hoi y te cua ban tai day..."
                    ),
                    label="Cau hoi",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Cai dat")
                context_input = gr.Textbox(
                    placeholder=(
                        "VD: Nam, 60 tuoi, tieu duong, "
                        "ngay 2 sau phau thuat (Tuy chon)"
                    ),
                    label=(
                        "Thong tin benh nhan "
                        "(Danh cho truy van theo ngu canh)"
                    ),
                    lines=3,
                )
                web_toggle = gr.Checkbox(
                    label="Bat tim kiem web Tavily",
                    value=True,
                )
                clear_btn = gr.Button("Xoa cuoc tro chuyen")

        def user_message(user_msg, chat_hist):
            chat_hist = chat_hist + [
                {"role": "user", "content": user_msg},
            ]
            return "", chat_hist

        def bot_response(chat_hist, context, use_web_feat):
            user_question = chat_hist[-1]["content"]
            bot_reply = respond(
                user_question,
                chat_hist[:-1],
                context,
                use_web_feat,
            )
            chat_hist = chat_hist + [
                {"role": "assistant", "content": bot_reply},
            ]
            return chat_hist

        msg_input.submit(
            user_message,
            [msg_input, chatbot],
            [msg_input, chatbot],
            queue=False,
        ).then(
            bot_response,
            [chatbot, context_input, web_toggle],
            chatbot,
        )

        clear_btn.click(
            lambda: [], None, chatbot, queue=False,
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
