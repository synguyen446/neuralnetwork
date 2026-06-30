class Node:
    def __init__(self, value, next=None, prev=None):
        self.next = next
        self.prev = prev
        self.value = value

    def return_value(self):
        return self.value


class LinkedList:
    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail

    def insert_head(self, node):
        if not self.head:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head = node

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.return_value()
            current = current.next
