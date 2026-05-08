def dictsum(dic, t):
    return sum([dic[key][t] for key in dic if t in dic[key]])