import streamlit as st
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import glob
from helper import draw_embed, create_spectrogram, read_audio, record, save_record
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import base64


st.set_page_config(
    page_title="Time Series Playground by EDHEC ", layout="wide", page_icon="./images/flask.png"
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()


st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

st.title("EDHEC - Time series playground 🧪")
st.subheader("Understand Time Series Models (ARIMA) with a Finance Example 📈")
#
# "# Streamlit showcase"
#
# model_load_state = st.text("Loading pretrained models...")
#
# seed = 42
# low_mem = False
# num_generated = 0
# enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
# syn_model_dir = Path("synthesizer/saved_models/logs-pretrained/")
# voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")
# encoder.load_model(enc_model_fpath)
# synthesizer = Synthesizer(
#     syn_model_dir.joinpath("taco_pretrained"), low_mem=low_mem, seed=seed
# )
# vocoder.load_model(voc_model_fpath)
#
# model_load_state.text("Loaded pretrained models!")
#
# st.header("1. Record your own voice")
#
# filename = st.text_input("Choose a filename: ")
#
# if st.button(f"Click to Record"):
#     if filename == "":
#         st.warning("Choose a filename.")
#     else:
#         record_state = st.text("Recording...")
#         duration = 5  # seconds
#         fs = 48000
#         myrecording = record(duration, fs)
#         record_state.text(f"Saving sample as {filename}.mp3")
#
#         path_myrecording = f"./samples/{filename}.mp3"
#
#         save_record(path_myrecording, myrecording, fs)
#         record_state.text(f"Done! Saved sample as {filename}.mp3")
#
#         st.audio(read_audio(path_myrecording))
#
#         fig = create_spectrogram(path_myrecording)
#         st.pyplot(fig)
#
# "## 2. Choose an audio record"
#
# audio_folder = "samples"
# filenames = glob.glob(os.path.join(audio_folder, "*.mp3"))
# selected_filename = st.selectbox("Select a file", filenames)
#
# if selected_filename is not None:
#     # Create embedding
#     in_fpath = Path(selected_filename.replace('"', "").replace("'", ""))
#     original_wav, sampling_rate = librosa.load(str(in_fpath))
#     preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#     embed = encoder.embed_utterance(preprocessed_wav)
#     st.success("Created the embedding")
#
#     st.audio(read_audio(in_fpath))
#
#     if st.sidebar.checkbox("Do you want to change your embedding?"):
#         height = int(np.sqrt(len(embed)))
#         shape = (height, -1)
#         matrix_embed = np.round(embed, 2).reshape(shape)
#         matrix_embed = [list(row) for row in matrix_embed]
#         a = st.text_area("Change your embedding:", value=str(matrix_embed).replace("],", "],\n"))
#
#         matrix = [[float(x) for x in row.strip("[] \n").split(",")] for row in a.split("],")]
#         embed = np.array(matrix).flatten()
#
#     fig = draw_embed(embed, "myembedding", None)
#     st.pyplot(fig)
#
#
# "## 3. Synthesize text."
# text = st.text_input("Write a sentence (+-20 words) to be synthesized:")
#
#
# def pgbar(i, seq_len, b_size, gen_rate):
#     mybar.progress(i / seq_len)
#
#
# if st.button("Click to synthesize"):
#     texts = [text]
#     embeds = [embed]
#
#     # generate waveform
#     with st.spinner("Generating your speech..."):
#         specs = synthesizer.synthesize_spectrograms(texts, embeds)
#         spec = specs[0]
#         synthesize_state = st.text("Created the mel spectrogram")
#         synthesize_state.text("Generating the waveform...")
#         mybar = st.progress(0)
#         generated_wav = vocoder.infer_waveform(spec, progress_callback=pgbar)
#         generated_wav = np.pad(
#             generated_wav, (0, synthesizer.sample_rate), mode="constant"
#         )
#         generated_wav = encoder.preprocess_wav(generated_wav)
#         synthesize_state.text("Synthesized the waveform")
#         st.success("Done!")
#
#     # Save it on the disk
#     filename = "demo_output_%02d.wav" % num_generated
#     sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
#     num_generated += 1
#     synthesize_state.text("\nSaved output as %s\n\n" % filename)
#     st.audio(read_audio(filename))

if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### 👨🏼‍💻 **App Contributors:** ")
st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison/app-predictive-analytics'}) 🚀 ")
st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made by ",
        link("https://www.edhec.edu/en", "EDHEC - Gaëtan Brison"),
        "👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()

