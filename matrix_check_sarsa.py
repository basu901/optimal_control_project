import numpy as np
import pandas as pd 
import random as rd 
import copy as cp
import test_plot as tp


def find_max_action(Q,directions,num,x,y,visited,R,dim):
	Q_copy = cp.deepcopy(Q[num,:])
	#print(Q_copy)

	for a in directions:
		n_s = find_next_state(x,y,a,dim)
                not_visited_reward = (1-visited[n_s])*R
		Q_copy[a] +=not_visited_reward

	max_Q = max(Q_copy[directions])
	
	action_greedy = [i for i,j in enumerate(Q_copy) if j == max_Q and i in directions]

	del Q_copy

	return action_greedy


def epsilon_greedy(epsilon,Q,directions,num,x,y,visited,R,dim):
	r = np.random.random_sample()

	if r>epsilon:
		action_greedy = find_max_action(Q,directions,num,x,y,visited,R,dim)
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

'''
def calc_go(l):
	prod = 1
	for i in l.values():
		prod *= i
	return prod'''



def q_learn(epochs,alpha,gamma,epsilon,d_shape):
	Q = np.zeros(shape = (d_shape*d_shape,8)) #Cost matrix
	A = np.ones(8) #Array of possible directions
	visited = np.zeros(d_shape*d_shape)#array of visited states
	revisited = np.zeros(d_shape*d_shape)#array of states which are revisited
	
	R_visit = 10
	R_revisit = -20
	R_not_visited = 100

	R_movement = np.array([[0,-10,0,-20,-20,0,0,0,0],#0
		[-10,0,-10,-20,-20,-20,0,0,0],#1
		[0,-10,0,0,-20,-10,0,0,0],#2
		[-20,-20,0,0,-10,0,-10,-20,0],#3
		[-20,-20,-20,-10,0,-10,-20,-20,-20],#4
		[0,-20,-10,0,-10,0,0,-20,-20],#5
		[0,0,0,-10,-20,0,0,-10,0],#6
		[0,0,0,-20,-20,-20,-10,0,-10],#7
		[0,0,0,0,-20,-20,0,-10,0]])#8
	
	Q_record = np.zeros(shape=(epochs,d_shape*d_shape,8))

	action_record = np.zeros(shape = (epochs,d_shape*d_shape))

	present_x_total = list()
	next_x_total = list()
	action_x_total = list()

	for i in range(0,epochs):
		num = 0
		visited[num]=1
		present_x = list()
		next_x = list()
		action_x = list()
		x,y = calc_index(num,d_shape)
		present_x.append(num)
		directions = available_actions(Q,A,x,y,d_shape)
		action = epsilon_greedy(epsilon,Q,directions,num,x,y,visited,R_not_visited,d_shape)
		action_x.append(action)
		while not visited.all()==1:
			next_state = find_next_state(x,y,action,d_shape)
			x_next,y_next = calc_index(next_state,d_shape)
			next_state_directions = available_actions(Q,A,x_next,y_next,d_shape)
			next_state_action = epsilon_greedy(epsilon,Q,next_state_directions,next_state,x_next,y_next,visited,R_not_visited,d_shape)
			#print("Next State",next_state)
			if visited[next_state]==1:
                                revisited[next_state]==1
                        reward = R_movement[num,next_state] + ((1-visited[next_state])*R_visit) + (revisited[next_state]*R_revisit)
                        error = reward + gamma*(Q[next_state,next_state_action])-Q[num,action]
                        Q[num,action] = Q[num,action]+alpha*error
                        visited[next_state] = 1
                        num = next_state
                        x,y = calc_index(num,d_shape)
                        present_x.append(num)
                        directions = available_actions(Q,A,x,y,d_shape)
                        action = epsilon_greedy(epsilon,Q,directions,num,x,y,visited,R_not_visited,d_shape)
                        action_x.append(action)
			
                epsilon /= 1.0005
                Q_record[i,:,:] = Q
                present_x_total.append(present_x)
                next_x_total.append(next_x)
                action_x_total.append(action_x)
                visited[:] = np.zeros(d_shape*d_shape)
			
		del present_x
		del next_x
		del action_x
                max_val = final_action(Q)
                action_record[i] = max_val		
		 

	print("Present",present_x_total[-1])
	print("Aciton",action_x_total[-1])
	print("Next State",next_x_total[-1])
	print(Q_record[-1,:,:])
	print("Revisited::",revisited)
	final_output(Q)
	tp.action_plot(action_record)



def final_output(Q):
        ar = np.zeros(np.size(Q,0))

        for i in range(0,np.size(Q,0)):
                check = 1
                idx = 0
                for j in range(0,np.size(Q,1)):
                        if not Q[i,j]==0:
                                if check==1:
                                        maxm = Q[i,j]
                                        idx = j
                                        check = 0
                                else:
                                        if maxm<Q[i,j]:
                                                maxm = Q[i,j]
                                                idx = j
                print("Grid",i,"idx",idx)
                

def final_action(Q):
        ar = np.zeros(np.size(Q,0))
        for i in range(0,np.size(Q,0)):
                check = 1
                idx = 0
                for j in range(0,np.size(Q,1)):
                        if not Q[i,j]==0:
                                if check==1:
                                        maxm = Q[i,j]
                                        idx = j
                                        check = 0
                                else:
                                        if maxm < Q[i,j]:
                                                maxm = Q[i,j]
                                                idx = j
                ar[i] = idx
        return ar

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
	q_learn(20000,0.05,0.9,0.4,3)

