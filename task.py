#!/usr/bin/python3
import numpy as np
import numpy.linalg
import sys
import matplotlib.pyplot as pp




class Task():

	def compute_l(self,l,s):
		t1 = np.dot(np.dot(l,s),l.transpose().conjugate())
		t2 = np.dot(np.dot(l.transpose().conjugate(),l),s)
		t3 = np.dot(np.dot(s,l.transpose().conjugate()),l)
		res = np.matrix([[complex(0,0)] * self.size] * self.size)
		for i in range(self.size):
			for j in range(self.size):
				res[i,j] = t1[i,j] - 0.5 * t2[i,j] - 0.5 * t3[i,j]
		return res
	
	def init_settings2(self):
		self.size = 8 # Размерность системы
		self.w = 4 # Энергия фотоноы
		self.a = 0.7 # Амплитуды перехода фотонов
		self.connections = [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(5,6)] # Соединения между полостями
		self.sink_ind = 7 # Индекс ванны
		self.flow = [6] # Полости, из которых идёт сток в ванну
		self.dt = 0.025 # Квант времени
		self.iters = 4000 # Количество итераций
		self.start_pos = 0 # В какой позиции изначально назодится фотон
	
	def init_settings1(self):
		self.size = 8 # Размерность системы
		self.w = 4 # Энергия фотоноы
		self.a = 0.7 # Амплитуды перехода фотонов
		self.connections = [(0,1),(1,2),(1,3),(2,4),(3,4),(4,5),(5,6)] # Соединения между полостями
		self.sink_ind = 7 # Индекс ванны
		self.flow = [6] # Полости, из которых идёт сток в ванну
		self.dt = 0.025 # Квант времени
		self.iters = 4000 # Количество итераций
		self.start_pos = 0 # В какой позиции изначально назодится фотон

	def evol(self):
		time_moments = []
		sink_amp = []
		# h -- гамильтониан, l -- сток, s -- стартовая позиция
		h = np.matrix([[complex(0,0)] * self.size] * self.size)
		l = np.matrix([[complex(0,0)] * self.size] * self.size)
		s = np.matrix([[complex(0,0)] * self.size] * self.size)
	
		# заполняем
		for pair in self.connections:
			h[pair[0],pair[1]] = self.a
			h[pair[1],pair[0]] = self.a
		for i in range(self.size):
			if i != self.sink_ind:
				h[i,i] = self.w
		for item in self.flow:
			l[self.sink_ind,item] = self.a
		s[self.start_pos,self.start_pos] = 1
	
		# считаем унитарное преобразование
		eVal,eVec = np.linalg.eig(h)
		eVec1 = np.matrix([[complex(0,0)] * self.size] * self.size)
		for i in range(self.size):
			for j in range(self.size):
				eVec1[j,i] = eVec[i,j] * np.exp(complex(0,-self.dt*eVal[i]))
		u = np.dot(eVec1,eVec.conjugate())
		
		time_moments.append(0.0)
		sink_amp.append(s[self.sink_ind,self.sink_ind].real)
	
		for i in range(self.iters):
			s = np.dot(np.dot(u,s),u.conjugate())
			new_l = self.compute_l(l,s)
			
			for j in range(self.size):
				for k in range(self.size):
					s[j,k] += new_l[j,k]
			
			time_moments.append((i+1) * self.dt)
			sink_amp.append(s[self.sink_ind,self.sink_ind].real)
		print(sink_amp[-1])		
		return (time_moments,sink_amp)

	def graph(self):
		time_moments,sink_amp = self.evol()
		pp.plot(time_moments,sink_amp)
		pp.show()

	def graph2(self):
		self.init_settings1()
		time_moments1,sink_amp1 = self.evol()
		self.init_settings2()
		time_moments2,sink_amp2 = self.evol()
		pp.plot(time_moments1,sink_amp1,'r',time_moments2,sink_amp2,'g')
		pp.show()

if __name__ == "__main__":
	task = Task()
	task.graph2()

