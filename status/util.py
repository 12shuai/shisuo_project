import math
def getNorm(state):
    value=0
    for k,v in state.items():
        value+=v**2

    return math.sqrt(value)






