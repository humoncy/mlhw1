import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import scipy.linalg
import math
import sys

'''
PLU decomposition
P A = L U
P: permutation matrix
	propertity: P.inv() = P.transpose()
Therefore:
	A = P L U is the similar one.
Why not A = L U?
A can do decomposition only if the pivot is in the good place.
ex:
[ 0 1 0
  1 0 0
  0 0 1 ]
Must permute row(1) and row(2) to first so that it could be decomposed by L U
'''
def plu_decomposition(A):
	s = len(A)
	U = A
	L = np.zeros((s,s))
	P = np.eye(s)
	for j in range(s):
		# Pivot voting
		ind = np.argmax(abs(U[j:s, j]))
		ind += j
		if ind > j:
			# permute rows
			P[[j,ind],:] = P[[ind,j],:]
			U[[j,ind],:] = U[[ind,j],:]
			L[[j,ind],:] = L[[ind,j],:]
		# LU
		L[j, j] = 1
		for i in range(j+1, s):
			c = U[i, j] / U[j,j]
			U[i,:] -= U[j,:] * c
			L[i, j] = c
	
	return [P.transpose(), L, U]
	
'''
Calculate inverse of A by LU decomposition
P A = L U
A x = I -> L U x = I
suppose U x = d
then L d = I
Solve
	1. L d = I
	2. U x = d
to get x = inv(A)
'''
def inverseLU(phi, _lambda, num_bases):
	A = np.dot(np.transpose(phi), phi) + _lambda * np.identity(num_bases)
	#[p, l, u] = scipy.linalg.lu(A)
	[p, l, u] = plu_decomposition(A)
	x = np.dot(np.dot(np.linalg.pinv(u), np.linalg.pinv(l)), p.transpose())
	return x
	
'''
main
'''
if len(sys.argv) < 4:
	print("Usage:", sys.argv[0], "<file path>", "<num_polynomial_bases>", "<lambda>")
	sys.exit(1)
file_path = sys.argv[1]
num_bases = int(sys.argv[2])
_lambda = float(sys.argv[3])

if num_bases < 1:
	print("Number of bases should at least be 1.")
	sys.exit(1)

inputs = np.loadtxt(file_path, delimiter = ',')

x = inputs[:,0]
y = inputs[:,1]

phi = np.zeros((len(inputs),num_bases))
for i in range(num_bases):
	phi[:, i] = x**i
# print(phi)

# inverse not by LU decomposition
#w = np.dot(np.dot( inv( np.dot(np.transpose(phi), phi) + _lambda* np.identity(num_bases) ), np.transpose(phi)), np.transpose(y))
#print(w)

inv_lu_atalambdai = inverseLU(phi, _lambda, num_bases)
w = np.dot(np.dot( inv_lu_atalambdai, np.transpose(phi)), np.transpose(y))


b = np.dot(phi, w)

print("equation: ", end='')
for i in reversed(range(num_bases)):
	if i > 0:
		print("%.4f*x^%d + " % (w[i], i), end='')
	else:
		print('%.4f' % w[i])

error = norm(b-y)
print("error: %.4f" % error)

# plot
fig = plt.figure()
# draw input points
ax1 = fig.add_subplot(111)
ax1.scatter(x,y)

# draw function
l = np.linspace(min(x)-5, max(x)+5)
f = np.zeros(len(l))
for i in range(num_bases):
	f += w[i] * l**i
ax1.plot(l, f, 'b-')
plt.show()
