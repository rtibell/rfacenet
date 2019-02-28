
def extrFromList(lst):
    zlst = list(zip(*lst))
    return ([ min(l) for l in zlst ] , [ max(l) for l in zlst ]    
