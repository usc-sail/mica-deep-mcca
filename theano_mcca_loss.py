import theano
import theano.tensor as T
from scipy.io import loadmat
import numpy as np
from theano.tensor.slinalg import eigvalsh

def mcca_loss(N):
	'''
	N - number of modalities (>2)
	D - dimension of each modality
	main loss is wrapped into this function
	'''
	def inner_mcca_objective(y_true, y_pred):
		D = y_pred.shape[1]//N
		modality_range = [(D*i, (i+1)*D) for i in range(N)]
		#X = np.dstack((X_[:, i:j] for i,j in modality_range))
		m = y_pred.shape[0]
		#X = y_pred.T

		Xbar = y_pred.T - (1.0 / m) * T.dot(y_pred.T, T.ones([m, m]))
		X_Rw = (1.0 / (m-1)) * T.dot(Xbar, Xbar.T)
		Rw_ = T.zeros([D, D])
		Xsum = T.zeros([D, m])
		for i,j in modality_range:
			Rw_ = Rw_ + X_Rw[i:j, i:j]
			Xsum = Xsum + y_pred.T[i:j, :]
		Xmean = Xsum/N
		# total cov
		Xmean_bar = Xmean - (1.0 / m) * T.dot(Xmean, T.ones([m, m]))
		Rt_ = ((N * N * 1.0) / (m-1)) * T.dot(Xmean_bar, Xmean_bar.T) 
		Rb_ = (Rt_ - Rw_)/(N - 1)

		# -- just simple regularization: Rw_ = Rw_ + r1 * T.eye(D)
		# shrinkage regularize - gamma
		Rw_reg_ = ((1 - gamma)*Rw_) + (gamma*(Rw_.diagonal().mean())*T.eye(D))
		ISC_ = eigvalsh(Rb_, Rw_reg_)

		l = T.nlinalg.eigh(Rw_reg_, 'L')

		# do Cholesky to do Generalized Eig Problem
		L = T.slinalg.cholesky(Rw_reg_)
		C_ = T.dot(T.nlinalg.matrix_inverse(L), Rb_)
		C = T.dot(C_, T.nlinalg.matrix_inverse(L).T)
		C_eigval, C_eigvec = T.nlinalg.eig(C)
		indx_ = T.argsort(C_eigval)[::-1]
		W_ = T.dot(T.nlinalg.matrix_inverse(L).T, C_eigvec)[:, indx_]
		d_ = T.diag(1.0/T.sqrt((W_*W_).sum(axis=0)))
		W_ = T.dot(W_, d_)
		# recompute ISC
		ISC = T.diag(T.dot(T.dot(W_.T, Rb_), W_)) / T.diag(T.dot(T.dot(W_.T, Rw_), W_)) 
		corr = T.sqrt(T.sum(ISC*ISC))
		
		return -1*ISC[0]#-corr

	return inner_mcca_objective
