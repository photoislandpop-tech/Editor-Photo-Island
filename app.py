import os
import cv2
import io
import zipfile
import numpy as np
from PIL import Image
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Editor Photo Island", layout="wide")
st.title("📸 Editor Photo Island")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None

def preservar_alta_resolucion(imagen_original_pil, imagen_editada_pil):
    orig = cv2.cvtColor(np.array(imagen_original_pil), cv2.COLOR_RGB2BGR)
    edit = cv2.cvtColor(np.array(imagen_editada_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = orig.shape
    edit_resized = cv2.resize(edit, (w, h), interpolation=cv2.INTER_CUBIC)
    orig_lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)
    edit_lab = cv2.cvtColor(edit_resized, cv2.COLOR_BGR2LAB)
    resultado_lab = cv2.merge([edit_lab[:, :, 0], edit_lab[:, :, 1], edit_lab[:, :, 2]])
    resultado = cv2.cvtColor(resultado_lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))

prompt = """
Esta foto puede que tenga o no tenga los siguientes problemas: oscura, quemada, balance de blancos incorrecto; sensor de camara sucio lo cual podría causar puntos negros en el cielo. Podría tener el horizonte torcido. Editala para que quede lo más natural posible en cuanto a iluminación y colores, pero sin cambiar nada de detalles de rostros, poses, personas, escena etc. Solo exposición, colores etc. Y color de piel natural. No necesariamente todas las personas presentes en la foto deban tener el mismo color de piel entre ellos. De ser necesario, dar contrastes, saturaciones necesarias para que la foto tenga colores naturales, no exagerados ni opacos. La foto editada debe mantener la misma resolución de la original.
"""

uploaded_files = st.file_uploader("Sube las fotos (muchas a la vez)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files and client:
    editadas_para_zip = [] # aqui guardamos todo para el zip

    for file in uploaded_files:
        st.divider()
        original = Image.open(file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption=f"Original: {file.name}", use_container_width=True)

        with st.spinner(f"Editando {file.name}..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-image",
                    contents=[original, prompt],
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                        editada_temp = Image.open(io.BytesIO(part.inline_data.data))
                        final_hd = preservar_alta_resolucion(original, editada_temp)

                        with col2:
                            st.image(final_hd, caption="✅ Editada HD", use_container_width=True)

                        # Guardar para zip
                        buf = io.BytesIO()
                        final_hd.save(buf, format="JPEG", quality=95)
                        editadas_para_zip.append((f"HD_{file.name}", buf.getvalue()))

                        st.download_button(f"📥 Descargar {file.name}", buf.getvalue(), file_name=f"HD_{file.name}", mime="image/jpeg", key=file.name)
            except Exception as e:
                st.error(f"Error con {file.name}: {e}")

    # BOTON ZIP SI HAY MAS DE 1
    if len(editadas_para_zip) > 1:
        st.divider()
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for nombre, data in editadas_para_zip:
                zip_file.writestr(nombre, data)

        st.download_button(
            label=f"📦 Descargar TODO ({len(editadas_para_zip)} fotos) en ZIP",
            data=zip_buffer.getvalue(),
            file_name="fotos_editadas_HD.zip",
            mime="application/zip"
        )