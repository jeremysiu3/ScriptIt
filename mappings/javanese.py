# 1. Consonants (with inherent 'a')
JAVANESE_CONSONANTS = {
    'h': 'ꦲ',  # ha
    'n': 'ꦤ',  # na
    'c': 'ꦕ',  # ca
    'r': 'ꦫ',  # ra
    'k': 'ꦏ',  # ka
    'd': 'ꦢ',  # da
    't': 'ꦠ',  # ta
    's': 'ꦱ',  # sa
    'w': 'ꦮ',  # wa
    'l': 'ꦭ',  # la
    'p': 'ꦥ',  # pa
    'dh': 'ꦝ', # dha
    'j': 'ꦗ',  # ja
    'y': 'ꦪ',  # ya
    'ny': 'ꦚ', # nya
    'm': 'ꦩ',  # ma
    'g': 'ꦒ',  # ga
    'b': 'ꦧ',  # ba
    'th': 'ꦛ', # tha
    'ng': 'ꦔ', # nga
    'H': 'ꦲ',
    'N': 'ꦤ',
    'C': 'ꦕ',
    'R': 'ꦫ',
    'K': 'ꦏ',
    'D': 'ꦢ',
    'T': 'ꦠ',
    'S': 'ꦱ',
    'W': 'ꦮ',
    'L': 'ꦭ',
    'P': 'ꦥ',
    'DH': 'ꦝ',
    'Dh': 'ꦝ',
    'J': 'ꦗ',
    'Y': 'ꦪ',
    'NY': 'ꦚ',
    'Ny': 'ꦚ',
    'M': 'ꦩ',
    'G': 'ꦒ',
    'B': 'ꦧ',
    'TH': 'ꦛ',
    'Th': 'ꦛ',
    'NG': 'ꦔ',
    'Ng': 'ꦔ',
}

# 2. Vowel diacritics - attach to consonants
JAVANESE_VOWEL_DIACRITICS = {
    'a': '',        # inherent
    'aa': 'ꦴ',      # taling tarung
    'i': 'ꦶ',       # wulu
    'ii': 'ꦷ',      # wulu melik
    'u': 'ꦸ',       # suku
    'uu': 'ꦹ',      # suku mendut
    'e': 'ꦺ',       # taling
    'o': 'ꦺꦴ',      # taling tarung
}

# 3. Independent vowels - used at word start
JAVANESE_INDEPENDENT_VOWELS = {
    'a': 'ꦄ',
    'i': 'ꦆ',
    'u': 'ꦈ',
    'e': 'ꦌ',
    'o': 'ꦎ',
}

# Pangkon (like halant) - removes inherent 'a'
JAVANESE_PANGKON = '꧀'
