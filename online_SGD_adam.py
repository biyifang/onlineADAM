import numpy as np 
import math
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize

#normalize the data!

def find_j(w, sample, label,numberClass):
	label = int(label)
	for j in range(0,numberClass):
		temp = 0.0
		if j != label:
			temp = temp + 1.0
		temp = temp + np.dot(w[j],sample) - np.dot(w[label],sample)
		if j == 0:
			flag = 0
			largest = temp
		if temp > largest:
			largest = temp
			flag = j
	return flag

#def predict(w, sample):
#	for j in range(-1,2):
#		temp = 0.0
#		if j == -1:
#			flag = -1
#			temp = np.dot(w[j+1], sample)
#			largest = temp
#		else:
#			temp = np.dot(w[j+1], sample)
#			if temp > largest:
#				flag = j
#	return flag

def loss(w, j_star, sample, label):
	label = int(label)
	if j_star != label:
		loss_ind = 1.0
	else:
		loss_ind = 0.0
	loss_ind = loss_ind + np.dot(w[j_star], sample) - np.dot(w[label], sample)
	return loss_ind

def grad(sample, j_star, label, numberClass, numFeat):
	label = int(label)
	w = np.zeros((numberClass, numFeat))
	w[j_star] = sample
	w[label] = (-1.0)*sample
	return w


eps = 1e-16

print("before load")
#feat_raw, label = load_svmlight_file("/home/bfang/set1.train.txt")
feat_raw,label = load_svmlight_file("/Users/biyifang/Desktop/research/Quantization/data/Webscope_C14B/ltrc_yahoo/set1.train.txt")
#feat_raw, label = load_svmlight_file("/Users/biyifang/Downloads/mnist8m.scale")
#feat_raw = np.array(feat_raw)
#feat_raw, label = shuffle(feat_raw, label)
print("after load")
feat = feat_raw
labe = label

numFeat = feat.shape[1]
#13346
numRow = feat.shape[0]
print(numRow)
beta_1 = 0.8
beta_2 = 0.81
#haiyou 10,000 and 100,000
T = 100000
numberClass = 10
burningP = 50000
K = 50000
np.random.seed(0)
w = np.random.normal(0,1,[numberClass,numFeat])
m = np.zeros((numberClass, numFeat))
v = np.zeros((numberClass, numFeat))
v_hat = np.zeros((numberClass, numFeat))
#file = open("file2_adam_loss_T_"+str(T)+"_0_00001", "w")
file = open("file3_adam_loss_T_"+str(T), "w")
#file_error = open("file2_adam_error_T_"+str(T)+"_0_00001", "w")
file_error = open("file3_adam_error_T_"+str(T), "w")
#+"_shuffle2"

#scaler = StandardScaler(with_mean=False)
#scaler.fit(feat[0:burningP])
#scaler.transform(feat)
#feat = normalize(feat[0:burningP],norm='l1',axis=0)

stepsize = 1.0/T

loss_j = 0.0
loss_memory = np.zeros(T)
error = 0

for i in range(numRow):

	#if(((i-burningP)%K) == 0 and (i-burningP)>0):
		#feat = feat_raw
		#scaler.fit(feat[0:i])
		#scaler.transform(feat)
		#feat = normalize(feat_raw[0:i],norm='l1',axis=0)
		#print(i)
		#print("normalization finished")
	if i%500000 == 0:
		print(i)

	#sample = feat[i].toarray()
	#sample = sample[0]
	#sample = feat[i]
	sample = np.squeeze(np.asarray(feat[i].todense()))
	j_star = find_j(w, sample, label[i], numberClass)
	error = 0


	loss_i = loss(w, j_star, sample, label[i])


	if i < (T - 1) :
		loss_memory[i] = loss_i
	else:
		if i == (T - 1):
			loss_memory[i] = loss_i
			loss_p = sum(loss_memory)/(T*1.0)
			file.write(str(loss_p) +'\n')
		else:
			loss_p_temp = (sum(loss_memory[1:]) + loss_i)/(T*1.0)
			if loss_p_temp > loss_p:
				loss_p = loss_p_temp
			loss_memory[:-1] = loss_memory[1:]
			loss_memory[T-1] = loss_i

			#if (((i-burningP)%10000) == 0 and i > 0 and (i-burningP)>=0):
			file.write(str(loss_p)+'\n')



	if j_star != int(label[i]):
		error = 1
		g = grad(sample, j_star, label[i], numberClass, numFeat)
		m = beta_1*m + (1 - beta_1)*g
		v = beta_2*v + (1 - beta_2)*(g*g)
		v_hat = np.maximum(v,v_hat)
		adam_temp = np.zeros((numberClass, numFeat))
		index_adam = v_hat != 0
		adam_temp[index_adam] = stepsize/np.sqrt(v_hat[index_adam])
		w = w - adam_temp*m



	file_error.write(str(error)+'\n')

	#if (((i-burningP)%10000) == 0 and i > 0):
	#	print(i)
	#	print("start")
	#	for j in range(i):
	#		sample_j = feat[j].toarray()
	#		sample_j = sample_j[0]
	#		j_star_j = find_j(w, sample_j, label[j])
	#		loss_j = loss(w, j_star_j, sample_j, label[j])
	#		if j < (T - 1):
	#			loss_memory[j] = loss_j
	#		else:
	#			if j == (T - 1):
	#				loss_memory[j] = loss_j
	#				loss_p = sum(loss_memory)/(T*1.0)
	#			else:
	#				loss_p_temp = (sum(loss_memory[1:]) + loss_j)/(T*1.0)
	#				if loss_p_temp > loss_p:
	#					loss_p =loss_p_temp
	#				loss_memory[:-1] = loss_memory[1:]
	#				loss_memory[T-1] = loss_j
	#	print(i)
	#	print(loss_p)
	#	file.write(str(loss_p)+'\n')
	#	print("finish")






file.close()
file_error.close()







#feat = feat.getrow()


