from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Local .env (en Render se ignora si no existe)
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
VS_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

st.set_page_config(page_title="KA Legal (Vector Store OpenAI)", layout="wide")
st.title("⚖️ KA Legal (Societario) — Streamlit + OpenAI Vector Store")

# Validaciones
if not API_KEY:
    st.error("Falta OPENAI_API_KEY (ponelo en .env o en Environment Variables).")
    st.stop()

if not VS_ID:
    st.error("Falta OPENAI_VECTOR_STORE_ID (el id del vector store, ej: vs_...).")
    st.stop()

client = OpenAI(api_key=API_KEY)

# Estado conversacional
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Sos un asistente legal societario (Argentina). "
                "Respondé en español, con citas cuando existan en los documentos. "
                "Si no hay soporte documental suficiente, decilo claramente."
                "Responder solo en base a los documentos que tienes cargados en VS."
            ),
        }
    ]

# Render de historial
for m in st.session_state.messages:
    if m["role"] in ("user", "assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

user_text = st.chat_input("Escribí tu consulta legal…")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # 🔑 Esto es lo equivalente a: Model + Tool "societario vector store"
        # Usamos Responses API + file_search apuntando al vector store.
        resp = client.responses.create(
            model=MODEL,
            input=st.session_state.messages,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [VS_ID],
                }
            ],
        )

        # Texto final
        answer_text = resp.output_text or "(sin respuesta)"
        placeholder.markdown(answer_text)

        # (Opcional) Mostrar “citas” / resultados de búsqueda si vinieran
        # Nota: la estructura exacta puede variar según versión, esto lo dejo robusto.
        with st.expander("📌 Evidencia / resultados del vector store (debug)", expanded=False):
            try:
                # Busca items tipo file_search en la salida
                # y muestra lo más útil que haya.
                for item in resp.output:
                    # item suele ser dict-like / object-like
                    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
                    if data.get("type") == "file_search_call":
                        st.json(data)
            except Exception as e:
                st.write("No se pudieron mostrar detalles:", str(e))

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
