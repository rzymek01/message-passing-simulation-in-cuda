#!/usr/bin/python
"""
Test generator for Message Spreading Simulator.

Scenarios:
1) n-clique:
  even nodes always pass messages (odd - never)
2) n-clique:
  even nodes always pass messages and
  odd nodes pass messages after receiving ceil(n/2) messages
3) n-clique:
  even nodes always pass messages and
  consecutive odd nodes pass messages compositing a chain
  (first odd node pass messages after receiving ceil(n/2) messages,
  second odd node pass messages after receiving ceil(n/2)+1 messages and so on)

Output:
graph representing a human network

 		  | %d			      // #V
 x #V	| %f %f %f %f 	// v_h_i, G_0_i, G_max_i, v_d_i
 		  | %d			      // #E
 x #V	| %d [%d, ...]	// #edges of v_i [v_j, ...]
 		  | %d			      // source v_i
 		  | %d %d %d 		  // t_c, t_p, t_s
"""

import sys

if len(sys.argv) < 3:
  sys.exit('Usage: %s <scenario> <#V>' % sys.argv[0])

scenario = int(sys.argv[1])
N = int(sys.argv[2])
E = N * (N - 1)
vertices = range(0, N, 1)

print str(N)
# @todo: scenarios
if 1 == scenario:
  for i in vertices:
    if 0 == i % 2:
      print "1 1 10 2"
    else:
      print "-1 1 10 2"
else:
  sys.exit('Undefined scenario #%d' % scenario)

print str(E)

for i in vertices:
  edges = list(vertices)
  edges.remove(i)
  print str(N - 1) + ' ' + ' '.join(map(str, edges))

print "0"
print "3 30 3630"

