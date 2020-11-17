def xs128p(state0, state1):
    MAXSIZE=0xFFFFFFFFFFFFFFFF
    s1 = state0
    s0 = state1
    s1 ^= (s1 << 23) & MAXSIZE
    s1 ^= (s1 >> 17)
    s1 ^= s0
    s1 ^= (s0 >> 26)
    state0 = state1
    state1 = s1
    generated = (state0 + state1) & MAXSIZE

    return state0, state1, generated

def main():
    x = 1234567890
    y = 9876543210
    f = open("xorshift128plus.txt","w")
    for i in range(4000000):
        x,y,out=xs128p(x,y)
        f.write(str(out)+"\n")
if __name__ == '__main__':
    main()
