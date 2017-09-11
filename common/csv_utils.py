#!/usr/bin/python3
# -*- coding: utf-8 -*-
import csv
import os


def load_csv(filename):
    data = []
    isSuccess = False
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
        isSuccess = True
    if not isSuccess:
        raise IOError("file load failed!")
    return data


def save_csv(filename, data):
    isSuccess = False
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)
        isSuccess = True
    if not isSuccess:
        raise IOError("file save failed!")
