import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sacrebleu

# 1️⃣ Load model once.
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# 2️⃣ Streamlit setup.
st.set_page_config(page_title="🌸 ನಮ್ಮ Translator", layout="centered")

# 3️⃣ Small CSS tweak.
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
.stTextArea textarea {background-color: lavender !important; color: black !important; font-size: 18px !important; border-radius: 10px !important; border: 1px solid powderblue !important;}
.stButton>button {background-color: powderblue !important; color: black !important; font-size: 18px !important; border-radius: 10px !important; border: none !important; padding: 10px 25px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🌸 ನಮ್ಮ Translator</h1>", unsafe_allow_html=True)

# 4️⃣ Examples with references.
examples = [
    ("Hello, Good Morning", "ಹಲೋ, ಶುಭೋದಯ"),
    ("Have a great day ahead", "ಮುಂದಿನ ದಿನ ಚೆನ್ನಾಗಿರಲಿ"),
    ("I love learning new languages", "ನನಗೆ ಹೊಸ ಭಾಷೆಗಳನ್ನು ಕಲಿಯುವುದು ಇಷ್ಟ"),
    ("Where is the nearest hospital", "ನಿಕಟದ ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ"),
    ("This is a beautiful place", "ಇದು ಸುಂದರ ಸ್ಥಳವಾಗಿದೆ")
]
example_texts = [""] + [ex[0] for ex in examples]

selected_example = st.selectbox("Choose an example", example_texts)
user_input = st.text_area("Enter English text.", value=selected_example, height=120)

# Optional user-provided reference for BLEU.
user_reference = st.text_input("Optional: paste a reference Kannada sentence here for BLEU scoring.", value="")

# 5️⃣ Improved translate function.
def translate(text,
              num_beams=5,
              length_penalty=1.2,
              max_length=128,
              no_repeat_ngram_size=2,
              early_stopping=True,
              temperature=1.0,
              top_k=None,
              top_p=None):
    """
    Improved generation settings. Adjust params if you want to experiment.
    """
    if not text.strip():
        return ""

    # tokenization.
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)

   
    bos_id = None
    # Try common ways to access lang id.
    if hasattr(tokenizer, "lang_code_to_id") and "kan_Knda" in tokenizer.lang_code_to_id:
        bos_id = tokenizer.lang_code_to_id["kan_Knda"]
    else:
        # fallback: convert token to id if tokenizer contains it.
        try:
            bos_id = tokenizer.convert_tokens_to_ids("kan_Knda")
        except Exception:
            bos_id = None

    gen_kwargs = dict(
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping
    )

    # optional sampling tweaks.
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if temperature != 1.0:
        gen_kwargs["temperature"] = temperature

    if bos_id is not None:
        gen_kwargs["forced_bos_token_id"] = bos_id

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return translated

# 6️⃣ BLEU calculation helper.
def compute_bleu(reference, prediction):
    """
    reference: single string in Kannada.
    prediction: single string.
    Returns BLEU score (0-100).
    """
    if not reference.strip():
        return None
    # sacrebleu expects list of hypos and list of list of refs.
    bleu = sacrebleu.corpus_bleu([prediction], [[reference]])
    return float(bleu.score)

# 7️⃣ Translate button and display.
if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some English text.")
    else:
        # translate with improved settings.
        translated = translate(
            user_input,
            num_beams=6,
            length_penalty=1.0,
            max_length=128,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        st.markdown("### Translated Kannada Text")
        st.success(translated)

        # decide reference.
        reference = ""
        if user_reference.strip():
            reference = user_reference.strip()
        else:
            # if user selected an example, use its ref.
            for src, ref in examples:
                if user_input.strip() == src:
                    reference = ref
                    break

        if reference:
            bleu_score = compute_bleu(reference, translated)
            if bleu_score is not None:
                st.info(f"BLEU score (0-100): {bleu_score:.2f}")
            else:
                st.info("BLEU could not be computed.")
        else:
            st.info("No reference provided. BLEU not computed.")