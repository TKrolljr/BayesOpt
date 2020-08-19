def foo(x, keyword=[]):
    keyword.append('a')
    print(keyword)

def bar(x,foo):
    foo(x)

if __name__ == '__main__':
    def f(a):
        return foo(a, keyword=[1])
    bar(1.0, f)
    bar(1.0, f)
    bar(1.0, f)