def convert_text(text, mapping_dict):

    sorted_mapping = sorted(mapping_dict.items(), key=lambda x: len(x[0]), reverse=True)

    for key, value in sorted_mapping:
        text = text.replace(key, value)

    return text

def convert_abugida(text, consonants, vowel_diacritics, independent_vowels, halant, add_final_halant):
    
    result = []
    i = 0

    sorted_consonants = sorted(consonants.keys(), key=len, reverse=True)
    sorted_vowels = sorted(vowel_diacritics.keys(), key=len, reverse=True)
    sorted_indep_vowels = sorted(independent_vowels.keys(), key=len, reverse=True)

    last_was_consonant = False

    while i < len(text):
        matched = False

        #Handle spaces and punctuation
        if text[i] in ' .,!?:;':
            result.append(text[i])
            last_was_consonant = False
            i += 1
            continue

        #Try to match consonant
        for cons in sorted_consonants:
            if text[i:i+len(cons)] == cons:
                cons_char = consonants[cons]
                i += len(cons)
            
            #Look for ahead for vowel
                vowel_matched = False

                for vowel in sorted_vowels:
                    if i < len(text) and text[i:i+len(vowel)] == vowel:
                        if vowel == 'a':
                            result.append(cons_char)
                        else:
                            result.append(cons_char + vowel_diacritics[vowel])
                        i += len(vowel)
                        vowel_matched = True
                        last_was_consonant = False
                        break
            
                if not vowel_matched:
                    next_is_consonant = False
                    for next_cons in sorted_consonants:
                        if i < len(text) and text[i:i+len(next_cons)] == next_cons:
                            result.append(cons_char + halant)
                            next_is_consonant = True
                            last_was_consonant = False
                            break
                
                    if not next_is_consonant:
                        result.append(cons_char)
                        last_was_consonant = True
            
                matched = True
                break

        if matched:
            continue

        #Try to match independent vowel
        for vowel in sorted_indep_vowels:
            if text[i:i+len(vowel)] == vowel:
                result.append(independent_vowels[vowel])
                i += len(vowel)
                matched = True
                last_was_consonant = False
                break
        if not matched:
            i += 1
        
    if add_final_halant and last_was_consonant and result:
            result[-1] = result[-1] + halant


    return ''.join(result)

def convert_korean(text, initial_consonants, vowels, final_consonants, names_dict = None):

    if names_dict is None:
        names_dict = {}

    result = []
    text = text.lower()
    i = 0

    sorted_initials = sorted(initial_consonants.keys(), key=len, reverse=True)
    sorted_vowels = sorted(vowels.keys(), key=len, reverse=True)
    sorted_finals = sorted(final_consonants.keys(), key=len, reverse=True)
    sorted_names = sorted(names_dict.keys(), key=len, reverse=True)

    while i < len(text):
        if text[i] in ' .,?!:;':
            result.append(text[i])
            i += 1
            continue
        
        initial_match = None
        vowel_match = None
        final_match = None

        names_matched = False
        for name in sorted_names:
            if text[i:i+len(name)] == name:
                if i + len(name) >= len(text) or text[i + len(name)] in ' .,!?:;':
                    result.append(names_dict[name])
                    i += len(name)
                    names_matched = True
                    break
        
        if names_matched: 
            continue

        for initial in sorted_initials:
            if initial and text[i:i+len(initial)] == initial:
                initial_match = initial
                i += len(initial)
                break
        
        if initial_match is None:
            initial_match = ''

        for vowel in sorted_vowels:
            if vowel and i < len(text) and text[i:i+len(vowel)] == vowel:
                vowel_match = vowel
                i += len(vowel)
                break
        
        if vowel_match is None:
            i += 1
            continue

        final_match = ''
        for final in sorted_finals:
            if final and i < len(text) and text[i:i+len(final)] == final:
                next_is_vowel = False
                temp_i = i + len(final)
                for v in sorted_vowels:
                    if v and temp_i < len(text) and text[temp_i:temp_i+len(v)] == v:
                        next_is_vowel = True
                        break
                
                if not next_is_vowel:
                    final_match = final
                    i += len(final)
                    break
        
        initial_idx = initial_consonants[initial_match]
        vowel_idx = vowels[vowel_match]
        final_idx = final_consonants[final_match]
        
        syllable_code = 0xAC00 + (initial_idx * 588) + (vowel_idx * 28) + final_idx
        result.append(chr(syllable_code))

    return ''.join(result)



