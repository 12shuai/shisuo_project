
def inScopes(value,scopes):

    for index,(mini,maxi) in enumerate(scopes):
        if mini < value< maxi:
            return index

    return -1
