#!/usr/bin/python
"""
Test generator for Message Spreading Simulator.

Scenarios:
1) n-chain:
  simple chain, all nodes pass messages


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
E = 2 * (N - 1)
vertices = range(0, N, 1)

print str(N)

if 1 == scenario:
  for i in vertices:
    print "1 1 10 3"
else:
  sys.exit('Undefined scenario #%d' % scenario)

print str(E)

for i in vertices:
  if i == 0:
    print '1 ' + str(i + 1)
  elif 0 < i < N - 1:
    print '2 ' + str(i - 1) + ' ' + str(i + 1)
  else:
    print '1 ' + str(i - 1)

print "0"
print "3 30 3630"

