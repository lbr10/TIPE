
def inf(a,b) :
    return a < b


class Heap :

    def tasse(self,i) :
        n = len(self.heap) - 1
        if 2 * i + 1 >= n :
            self.percole(i)
        else :
            self.tasse(2*i)
            self.tasse(2*i+1)
            self.percole(i)

    def __init__(self,l=[],compare=inf) :
        self.heap = [len(l)] + l
        self.comp = compare
        self.tasse(1)

    def remonte(self,i) :
        if i // 2 > 0 :
            if self.comp(self.heap[i],self.heap[i//2]) :
                self.heap[i],self.heap[i//2] = self.heap[i//2],self.heap[i]
                self.remonte(i//2)

    def add(self,x) :
        self.heap.append(x)
        self.heap[0] += 1
        self.remonte(self.heap[0])

    def take(self) :
        self.heap[0] -= 1
        x = self.heap.pop()
        return x

    def percole(self,i) :
        if 2 * i + 1 >= len(self.heap) - 1 :
            if 2 * i < len(self.heap) - 1 :
                j = 2 * i
                if self.comp(self.heap[j],self.heap[i]) :
                    self.heap[i],self.heap[j] = self.heap[j],self.heap[i]
                    self.percole(j)
        else :
            if self.comp(self.heap[2*i],self.heap[2*i+1]) :
                j = 2 * i
            else :
                j = 2 * i + 1
            if self.comp(self.heap[j],self.heap[i]) :
                self.heap[i],self.heap[j] = self.heap[j],self.heap[i]
                self.percole(j)

    def take_min(self) :
        self.heap[1],self.heap[-1] = self.heap[-1],self.heap[1]
        x = self.take()
        self.percole(1)
        self.heap[0] -= 1
        return x
