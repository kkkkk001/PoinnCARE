aa_to_num = {
    'A': 1,  # Alanine
    'R': 2,  # Arginine
    'N': 3,  # Asparagine
    'D': 4,  # Aspartic acid
    'C': 5,  # Cysteine
    'E': 6,  # Glutamic acid
    'Q': 7,  # Glutamine
    'G': 8,  # Glycine
    'H': 9,  # Histidine
    'I': 10, # Isoleucine
    'L': 11, # Leucine
    'K': 12, # Lysine
    'M': 13, # Methionine
    'F': 14, # Phenylalanine
    'P': 15, # Proline
    'S': 16, # Serine
    'T': 17, # Threonine
    'W': 18, # Tryptophan
    'Y': 19, # Tyrosine
    'V': 20, # Valine
    'U': 21, # Selenocysteine
    'O': 22, # Pyrrolysine
    'B': 23, # Asparagine or Aspartic acid
    'Z': 24, # Glutamine or Glutamic acid
    'J': 25, # Leucine or Isoleucine
    'X': 26, # Unknown
    '<mask>': 27
}



num_to_aa = {v: k for k, v in aa_to_num.items()}