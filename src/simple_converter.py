def convert_text(text, mapping_dict):

    sorted_mapping = sorted(mapping_dict.items(), key=lambda x: len(x[0]), reverse=True)

    for key, value in sorted_mapping:
        text = text.replace(key, value)

    return text

def convert_javanese(text, consonants, vowel_diacritics, independent_vowels, halant):
    
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


    return ''.join(result)


