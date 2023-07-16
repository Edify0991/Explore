from queue import Queue
q=Queue()  #初始化定义一个，maxsize==0
q.put(1)
q.put("yxf")
q.put(3)
print(q.empty())  #不为空  False
print(q.full())   #队列没有满  False
print(list(q.queue))
print(q.maxsize)  #maxsize在定义的时候，没有定义就默认为0，表示没有长度限制
print(q.qsize())  #此queue的长度

print(q.get())  #queue默认从头开始删除data，把1删除了
print(list(q.queue))
for i in list(q.queue):
    print(i)