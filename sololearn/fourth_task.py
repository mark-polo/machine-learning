S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

def G(data):
    g = len(data)
    l = data.count(1)

    return (2 * l/g * (1 - (l/g))) * g/len(S)

print( round(G(S) - G(A) - G(B), 5) )