#!/usr/bin/env python3
import argparse
import urllib.request

URL = 'https://dl.fbaipublicfiles.com/SymbolicMathematics/models/ode2.pth'
p = argparse.ArgumentParser()
p.add_argument('--output', default='ode2.pth')
a = p.parse_args()
print('Downloading official checkpoint:')
print(URL)
urllib.request.urlretrieve(URL, a.output)
print('Saved to', a.output)
