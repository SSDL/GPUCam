import numpy as np
import pdb

pos=np.array([[12., 23.], [12., 48.], [0., 0.], [0., 23.]])

filtered=-np.ones((1,2))
for i, el in enumerate(pos):
	if el[0] not in filtered[:,0] and el[1] not in filtered[:,1]: filtered=np.vstack((filtered, pos[i]))
pos=filtered[1:]
pdb.set_trace()
lines[0]=[lines[0][i] for i in pos[:,0]]
lines[1]=[lines[1][i] for i in pos[:,1]]
pdb.set_trace()

