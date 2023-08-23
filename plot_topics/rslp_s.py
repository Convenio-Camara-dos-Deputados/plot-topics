_RSLPS_EXCEP = {"lápis", "cais", "mais", "crúcis", "biquínis", "pois", "depois", "dois", "leis"}
_RSLPS_EXCEP_S = {
    "aliás",
    "pires",
    "lápis",
    "cais",
    "mais",
    "mas",
    "menos",
    "férias",
    "fezes",
    "pêsames",
    "crúcis",
    "gás",
    "atrás",
    "moisés",
    "através",
    "convés",
    "ês",
    "país",
    "após",
    "ambas",
    "ambos",
    "messias",
}


def stem(word: str) -> str:
    new_word = list(word)

    if len(word) < 3:
        return word

    if new_word[-1] == "s" and new_word[-2] == "n":
        new_word[-2] = "m"
        sing = "".join(new_word)
        sing = sing[:-1]
        return sing

    if new_word[-1] == "s" and new_word[-2] == "e" and new_word[-3] == "õ":
        new_word[-3] = "ã"
        new_word[-2] = "o"
        sing = "".join(new_word)
        sing = sing[:-1]
        return sing

    if new_word[-1] == "s" and new_word[-2] == "e" and new_word[-3] == "ã":
        if word == "mães":
            word = word[:-1]
            return word

        new_word[-2] = "o"
        sing = "".join(new_word)
        sing = sing[:-1]
        return sing

    if new_word[-1] == "s" and new_word[-2] == "i" and new_word[-3] == "a":
        if word != "cais" and word != "mais":
            new_word[-2] = "l"
            sing = "".join(new_word)
            sing = sing[:-1]
            return sing

    if new_word[-1] == "s" and new_word[-2] == "i" and new_word[-3] == "é":
        new_word[-3] = "e"
        new_word[-2] = "l"
        sing = "".join(new_word)
        sing = sing[:-1]
        return sing

    if new_word[-1] == "s" and new_word[-2] == "i" and new_word[-3] == "e":
        new_word[-3] = "e"
        new_word[-2] = "l"
        sing = "".join(new_word)
        sing = sing[:-1]
        return sing

    if new_word[-1] == "s" and new_word[-2] == "i" and new_word[-3] == "ó":
        new_word[-3] = "o"
        new_word[-2] = "l"
        sing = "".join(new_word)
        sing = sing[:-1]
        return sing

    if new_word[-1] == "s" and new_word[-2] == "i":
        if word not in _RSLPS_EXCEP:
            new_word[-1] = "l"
            sing = "".join(new_word)
            return sing

    if new_word[-1] == "s" and new_word[-2] == "e" and new_word[-3] == "l":
        word = word[:-2]
        return word

    if new_word[-1] == "s" and new_word[-2] == "e" and new_word[-3] == "r":
        word = word[:-2]
        return word

    if new_word[-1] == "s":
        if word not in _RSLPS_EXCEP_S:
            word = word[:-1]

    return word
