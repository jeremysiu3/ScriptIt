import streamlit as st
from mappings.russian import LATIN_TO_CYRILLIC
from mappings.greek import LATIN_TO_GREEK
from mappings.javanese import JAVANESE_CONSONANTS, JAVANESE_VOWEL_DIACRITICS, JAVANESE_INDEPENDENT_VOWELS, JAVANESE_PANGKON
from src.simple_converter import convert_text, convert_javanese

st.markdown("""
    <style>
    /* Target disabled text areas and make text white */
    textarea:disabled {
        color: white !important;
        -webkit-text-fill-color: white !important;
        opacity: 1 !important;
    }
    
    /* Make all labels white */
    label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("ScriptIt!")
st.info("This program is machine learning assisted and therefore will not be 100% accurate.")

language = st.selectbox("Select Language", ["Bengali", "Greek", "Hindi", "Javanese", "Korean", "Mandarin", "Russian"])

placeholder = ""
if language == "Mandarin":
    placeholder = "Use this format: ni3 hao3 (pinyin with tone numbers)"

input = st.text_area("Enter text in Latin letters:", height=150, placeholder=placeholder)

if input:
    if language == "Russian":
        output = convert_text(input, LATIN_TO_CYRILLIC)

    elif language == "Greek":
        output = convert_text(input, LATIN_TO_GREEK)

    elif language == "Hindi":
        from src.hindi.hindi_ml_converter import predict_vowels_hybrid
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        ml_output = predict_vowels_hybrid(input)
        output = transliterate(ml_output, sanscript.ITRANS, sanscript.DEVANAGARI)
        output = output.replace('्।', '।')  # Before punctuation
        output = output.replace('् ', ' ')   # Before space
        if output.endswith('्'):             # At end
            output = output[:-1]

    elif language == "Bengali":
        from src.bengali.bengali_ml_converter import predict_vowels_hybrid
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    
        # ML predicts vowel lengths
        ml_output = predict_vowels_hybrid(input)
        # Convert ITRANS → Bengali script
        output = transliterate(ml_output, sanscript.ITRANS, sanscript.BENGALI)
        # Post-process: Remove word-final halants
        output = output.replace('্।', '।')
        output = output.replace('্ ', ' ')
        if output.endswith('্'):
            output = output[:-1]

    elif language == "Javanese":
        output = convert_javanese(input, JAVANESE_CONSONANTS, JAVANESE_VOWEL_DIACRITICS, JAVANESE_INDEPENDENT_VOWELS, JAVANESE_PANGKON)
    
    elif language == "Korean":
        from src.korean.korean_ml_converter import convert_korean_ml
        output = convert_korean_ml(input)   

    elif language == "Mandarin":
        from src.mandarin.mandarin_ml_converter import convert_mandarin
        output = convert_mandarin(input)

    st.text_area("Converted text:", value=output, height=150, disabled=True)
