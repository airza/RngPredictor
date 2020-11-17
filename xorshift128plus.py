def xs128p():
    x = 1234567890
    y = 9876543210
    MAXSIZE=0xFFFFFFFFFFFFFFFF
    def _rand():
        nonlocal y,x
        s0,s1=y,x
        s1 ^= (s1 << 23) & MAXSIZE
        s1 ^= (s1 >> 17)
        s1 ^= s0
        s1 ^= (s0 >> 26)
        x = y
        y = s1
        generated = (x+y) & MAXSIZE
        return generated
    return _rand 

def main():
    xs = xs128p()
    f = open("xorshift128plus.txt","w")
    for i in range(4000000):
        f.write(str(xs())+"\n")
if __name__ == '__main__':
    main()
