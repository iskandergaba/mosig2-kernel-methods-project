import string


def label_encode(seq, args):
    code = args[0]
    for key, value in code.items():
        seq = seq.replace(key, value)
    return seq


def one_hot_encode(seq, args=[string.ascii_lowercase]):
    alphabet = args[0]
    vector = []
    for char in seq:
        temp = ''
        for letter in alphabet:
            temp += '1' if char == letter else '0'
        vector.append(temp)
    return vector
