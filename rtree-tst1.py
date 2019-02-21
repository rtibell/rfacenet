from rtree import index

left, bottom, right, top = (0.0, 0.0, 1.0, 1.0)

idx = index.Index()
idx.insert(0, (left, bottom, right, top))

a=list(idx.intersection((1.0, 1.0, 2.0, 2.0)))
print(a)
a=list(idx.intersection((1.0000001, 1.0000001, 2.0, 2.0)))
print(a)
a=list(idx.nearest((1.0000001, 1.0000001, 2.0, 2.0), 1))
print(a)

left, bottom, right, top = (0.0, 0.0, 1.0, 1.0)
idx.insert(1, (left-0.1, bottom-0.1, right+0.2, top+0.2))

a=list(idx.intersection((0.1, 0.1, 2.2, 2.2)))
print(a)
a=list(idx.intersection((1.0, 1.0, 2.0, 2.0)))
print(a)
a=list(idx.intersection((1.0000001, 1.0000001, 2.0, 2.0)))
print(a)
a=list(idx.nearest((1.0000001, 1.0000001, 2.0, 2.0), 1))
print(a)
