#!/usr/bin/python3
# -*- coding: utf-8 -*-


class Heap(object):
    '''
    max heap
      >>left child>>>>  >>left child>>>>
      ↑              ↓  ↑              ↓
    --↑--------------↓--↑--------------↓------------------------
    | root 6|****5****|****4****|*****3****|****2****|****1****|
    --↓--------------↓-------↑--↓----------------↑-----↑---------
      ↓              >>>>>>>>↑>>↓>>right child>>>>     ↑
      >>>>>>>right child>>>>>>  >>>>>left child>>>>>>>>

      min heap
      >>left child>>>>  >>left child>>>>
      ↑              ↓  ↑              ↓
    --↑--------------↓--↑--------------↓-----------------------
    | root 1|****2****|****3****|****4****|****5****|****6****|
    --↓--------------↓-------↑--↓----------------↑-----↑-------
      ↓              >>>>>>>>↑>>↓>>right child>>>>     ↑
      >>>>>>>right child>>>>>>  >>>>>left child>>>>>>>>

    type --the heap type,min_heap or max_heap
    element_seq --the data sequence for heap initialized,if the capacity is not -1
              use element_seq[:capacity],otherwise use whole element_seq
    capacity -- the heap capacity, -1 means define by system
    '''

    def __init__(self, type="min_heap", element_seq=[], capacity=-1):
        self.__elements = capacity != - \
            1 and element_seq[:capacity] or list(element_seq)
        self.__type = type
        self.__capacity = capacity
        self.__build_heap()

    def __build_heap(self):
        self.heap_sort()

    def push(self, data):
        size = len(self.__elements)
        index = size
        if size == self.__capacity:
            self.__elements[0] = data
            index = 0
        else:
            self.__elements.append(data)
        if self.__type == "min_heap":
            while index >= 0:
                self.adjust_min_heap(index)
                index = (index - 1) // 2
        else:
            while index >= 0:
                self.adjust_max_heap(index)
                index = (index - 1) // 2

    def heap_type(self):
        return self.__type

    def top(self):
        return self.__elements[0]

    def size(self):
        return len(self.__elements)

    def pop(self):
        ret = self.__elements[0]
        self.__elements[0] = self.__elements[-1]
        self.__elements = self.__elements[:-1]
        if self.__type == "min_heap":
            self.adjust_min_heap(0)
        else:
            self.adjust_max_heap(0)
        return ret

    def elements(self):
        return self.__elements

    def adjust_max_heap(self, index):
        end = len(self.__elements) - 1
        tmp = self.__elements[index]
        while index < end:
            child = 2 * index + 1
            if child >= end:
                break
            if self.__elements[child] < self.__elements[child + 1]:
                child = child + 1
            if tmp < self.__elements[child]:
                self.__elements[index] = self.__elements[child]
            else:
                break
            index = child
        self.__elements[index] = tmp

    def adjust_min_heap(self, index):
        end = len(self.__elements) - 1
        tmp = self.__elements[index]
        while index < end:
            child = 2 * index + 1
            if child >= end:
                break
            if self.__elements[child] > self.__elements[child + 1]:
                child = child + 1
            if tmp > self.__elements[child]:
                self.__elements[index] = self.__elements[child]
            else:
                break
            index = child
        self.__elements[index] = tmp

    def heap_sort(self):
        size = len(self.__elements)
        index = size - 1
        if self.__type == "min_heap":
            while index >= 0:
                self.adjust_min_heap(index)
                index = index - 1
        else:
            while index >= 0:
                self.adjust_max_heap(index)
                index = index - 1
