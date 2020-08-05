import numpy as np
import pandas as pd 
import random as rd 
import copy as cp
import operator


def find_max_action(Q,directions,num,x,y,visited,R,dim,not_allowed):
	Q_copy = cp.deepcopy(Q[num,:])
	#print(Q_copy)

	for a in directions:
		n_s = find_next_state(x,y,a,dim)
		if n_s not in not_allowed:
			not_visited_reward = (1-visited[n_s])*R
			Q_copy[a] +=not_visited_reward

	max_Q = max(Q_copy[directions])
	
	action_greedy = [i for i,j in enumerate(Q_copy) if j == max_Q and i in directions]

	del Q_copy

	return action_greedy


def epsilon_greedy(epsilon,Q,directions,num,x,y,visited,R,dim,not_allowed):
	r = np.random.random_sample()

	if r>epsilon:
		action_greedy = find_max_action(Q,directions,num,x,y,visited,R,dim,not_allowed)
		a = rd.choice(action_greedy)
	else:
		a = rd.choice(directions)
	return a


def available_actions(Q,C,x,y,n):
	directions = list()
	A = cp.deepcopy(C)
	if y==0:
		A[0] = 0
		A[3] = 0
		A[5] = 0
	if y==n-1:
		A[2] = 0
		A[4] = 0
		A[7] = 0
	if x==0:
		A[0] = 0
		A[1] = 0
		A[2] = 0
	if x==n-1:
		A[5] = 0
		A[6] = 0
		A[7] = 0

	#print(A)

	for i in range(0,np.size(A)):
		if not A[i]==0:
			directions.append(i)
	del A

	return directions


def calc_go(l):
	prod = 1
	for i in l.values():
		prod *= i
	return prod



def q_learn(epochs,alpha,gamma,epsilon,d_shape):
	Q = np.zeros(shape = (d_shape*d_shape,8)) #Cost matrix
	A = np.ones(8) #Array of possible directions
	visited = dict() #array of visited states
	revisited = dict()#array of states which are revisited
	
	R_visit = 10
	R_revisit = -20
	R_not_visited = 30
	
	not_allowed = [6,7,8,11,16,17,18]

	for i in range(0,(d_shape*d_shape)):
		if i in not_allowed:
			continue
		else:
			visited[i] = 0
			revisited[i] = 0

	R_movement = np.array([[0,-1,0,0,0,-1,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#0
		[-1,0,-1,0,0,-1,-100,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#1
		[0,-1,0,-1,0,0,-100,-100,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#2
		[0,0,-1,0,-1,0,0,-100,-100,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#3
		[0,0,0,-1,0,0,0,0,-100,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#4
		[-1,-1,0,0,0,0,-100,0,0,0,-1,-100,0,0,0,0,0,0,0,0,0,0,0,0,0],#5
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#6
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#7
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#8
		[0,0,0,-1,-1,0,0,0,-100,0,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0],#9
		[0,0,0,0,0,-1,-100,0,0,0,0,-100,0,0,0,-1,-100,0,0,0,0,0,0,0,0],#10
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#11
		[0,0,0,0,0,0,-100,-100,-100,0,0,-100,0,-1,0,0,-100,-100,-100,0,0,0,0,0,0],#12
		[0,0,0,0,0,0,0,-100,-100,-1,0,0,-1,0,-1,0,0,-100,-100,-1,0,0,0,0,0],#13
		[0,0,0,0,0,0,0,0,-100,-1,0,0,0,-1,0,0,0,0,-100,-1,0,0,0,0,0],#14
		[0,0,0,0,0,0,0,0,0,0,-1,-100,0,0,0,0,-100,0,0,0,-1,-1,0,0,0],#15
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#16
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#17
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#18
		[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,-100,0,0,0,0,-1,-1],#19
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-100,0,0,0,0,-1,0,0,0],#20
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-100,-100,0,0,-1,0,-1,0,0],#21
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-100,-100,-100,0,0,-1,0,-1,0],#22
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-100,-100,-1,0,0,-1,0,-1],#23
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-100,-1,0,0,0,-1,0]])#24

	
	Q_record = np.zeros(shape=(epochs,d_shape*d_shape,8))

	present_x_total = list()
	next_x_total = list()
	action_x_total = list()

	for j in range(0,epochs):
		num = 0
		visited[num]=1
		present_x = list()
		next_x = list()
		action_x = list()
		go = 0
		while go==0:
			x,y = calc_index(num,d_shape)
			present_x.append(num)
			directions = available_actions(Q,A,x,y,d_shape)
			#print(directions)
			action = epsilon_greedy(epsilon,Q,directions,num,x,y,visited,R_not_visited,d_shape,not_allowed)
			#print("Action taken",action)
			action_x.append(action)
			next_state = find_next_state(x,y,action,d_shape) 
			#print("Next State",next_state)
			next_x.append(next_state)
			if next_state in not_allowed:
				reward = R_movement[num,next_state] 
				error = reward 	+ gamma*max(Q[num,:])-Q[num,action]
				Q[num,action] = Q[num,action] + alpha*error
				for i in range(0,(d_shape*d_shape)):
					if i in not_allowed:
						continue
					else:
						visited[i] = 0
						revisited[i] = 0
				num = 0
				go = calc_go(visited)
				present_x.clear()
				action_x.clear()
				next_x.clear()
				continue

			else:
				if visited[next_state]==1:
					revisited[next_state]+=1
				reward = R_movement[num,next_state] + ((1-visited[next_state])*R_visit) 
				#print("Reward",reward)
				if not revisited[next_state]==0:
					reward += + R_revisit
				error = reward 	+ gamma*max(Q[next_state,:])-Q[num,action]
				Q[num,action] = Q[num,action] + alpha*error
				visited[next_state] = 1
				num = next_state
				go = calc_go(visited)
			
		if j%100==0:
			print(j)
		if not j==epochs-1:
			for i in range(0,(d_shape*d_shape)):
				if i in not_allowed:
					continue
				else:
					visited[i] = 0
					revisited[i] = 0
		Q_record[j,:,:] = Q
		present_x_total.append(present_x)
		next_x_total.append(next_x)
		action_x_total.append(action_x)
			
		del present_x
		del next_x
		del action_x

		
		 

	print("Present",present_x_total[-1])
	print("Aciton",action_x_total[-1])
	print("Next State",next_x_total[-1])
	print(Q_record[-1,:,:])
	print("Revisited::",revisited)
	final_output(Q,revisited,not_allowed)



def final_output(Q,revisited,not_allowed):
	Q_copy = cp.deepcopy(Q)
	for k,v in revisited.items():
		if not k in not_allowed:
			if not v==0:
				ar = remove_zeros(Q_copy[k],0)
				sorted_ar = sorted(ar.items(), key=operator.itemgetter(1),reverse = True)
				while not (revisited[k]+1==0):
					if sorted_ar:
						print("Grid",k,"idx", sorted_ar[0][0])
						del sorted_ar[0]
						revisited[k] -=1
					else:
						break
			else:
				ar = remove_zeros(Q_copy[k],0)
				sorted_ar = sorted(ar.items(), key=operator.itemgetter(1),reverse = True)
				print("Grid",k," idx",sorted_ar[0][0])


def remove_zeros(ar,val):
	map = dict()
	for i in range(0,len(ar)):
		if not ar[i]==val:
			map[i] = ar[i]
	return map

def find_next_state(x,y,action,d_shape):
	if action==0:
		next  = d_shape*(x-1)+(y-1)
	elif action==1:
		next = d_shape*(x-1)+y
	elif action==2:
		next = d_shape*(x-1)+(y+1)
	elif action==3:
		next = d_shape*x + (y-1)
	elif action==4:
		next = d_shape*x + (y+1)
	elif action==5:
		next = d_shape*(x+1) + (y-1)
	elif action==6:
		next = d_shape*(x+1) + y
	else:
		next = d_shape*(x+1) + (y+1)
	return next	


def calc_index(num,d_shape):
	x = int(num/d_shape)
	y = num%d_shape
	return x,y

if __name__=="__main__":
	q_learn(5000,0.06,0.9,0.4,5)

