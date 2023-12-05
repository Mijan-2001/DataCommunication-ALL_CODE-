import numpy as np
from matplotlib import pyplot as plt

# UNIPOLER.................................
# data = np.random.randint(0,2,25)
data = np.array([0,0,1,0,1,1,0,0,1,0,1,0,0,0,1,0])
time = np.arange(len(data))

plt.subplot(2, 3, 1)
plt.step(time, data,where='post')
plt.title('Unipolar')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)
plt.grid(True)
plt.yticks([-2,-1,0,1,2,3])
plt.xticks(time)



# NRZ-L.......................................
# data = np.random.randint(0,2,10)
# time = np.arange(len(data))
signal = np.zeros(len(data), dtype = int)

for i in range(len(data)):
    if data[i] == 0:
        signal[i] = -1
    else:
        signal[i] = 1
    
plt.subplot(2, 3, 2)
plt.step(time, signal,where='post')
plt.title('NRZ-L')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.text(0, 2, data)
plt.grid(True)
plt.yticks([-2,-1,0,1,2,3])
plt.xticks(time)


# NRZ-I.......................................
# data = np.random.randint(0,2,10)
# time = np.arange(len(data))

signal = np.zeros(len(data), dtype = int)
flg = True

for i in range(len(data)):
    if data[i] == 1:
        flg = not flg
    if flg:
        signal[i] = 1
    else:
        signal[i] = -1
        
plt.subplot(2, 3, 3)
plt.step(time, signal,where='post')
plt.title('NRZ-I')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

# plt.text(3, 8)
plt.grid(True)
plt.yticks([-2,-1,0,1,2,3])
plt.xticks(time)



# RZ...........................................
# data = np.random.randint(0,2,10)
time = np.linspace(0,len(data),len(data)*2)
signal = np.zeros(2*len(data), dtype = int)

for i in range(0,2*len(data),2):
    if data[i//2] == 1:
       signal[i] = 1
    else:
        signal[i] = -1
    signal[i+1] = 0
    
plt.subplot(2, 3, 4)
plt.step(time, signal,where='post')
plt.title('RZ')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

# plt.text(3, 8)
plt.grid(True)
plt.yticks([-2,-1,0,1,2,3])
plt.xticks(np.arange(len(data)))



# MANCHESTER.........................................
# data = np.random.randint(0,2,10)
time_org = np.arange(len(data))
signal = np.zeros(len(data)*2, dtype = int)

for i in range(0,len(data)*2,2):
    if data[i//2] == 0:
        signal[i] = 1
        signal[i+1] = -1
    else:
        signal[i] = -1
        signal[i+1] = 1


print(signal)
time = np.arange(len(signal))
plt.subplot(2, 3, 5)
plt.step(time, signal,where='post')
plt.title('Manchestor')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

plt.grid(True)
plt.yticks([-2,-1,0,1,2,3])
plt.xticks(time_org*2,time_org)



# DIFFERENTIAL MANCHESTER...............................
# data = np.random.randint(0,2,10)
time_org = np.arange(len(data))
signal = np.zeros(len(data)*2, dtype = int)

start = False
for i in range(0,len(data)*2,2):
    if start:
        if data[i//2] == 0:
            signal[i] =  -1*signal[i-1]
            signal[i+1] = signal[i-1]
        else:
            signal[i] = signal[i-1]
            signal[i+1] = -1* signal[i-1]
    else:
        start = True
        if data[i//2] == 0:
            signal[i] = -1
            signal[i+1] = 1
        else:
            signal[i] = 1
            signal[i+1] = -1


print(signal)
time = np.arange(len(signal))
plt.subplot(2, 3, 6)
plt.step(time, signal,where='post')
plt.title('Differential Manchestor')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

plt.grid(True)
plt.yticks([-2,-1,0,1,2,3])
plt.xticks(time_org*2,time_org)
plt.subplots_adjust(hspace=1)
plt.show()

# D to A.....................................................................
#AMPLITUDE SHIFT KEYING.....................
F1=10 
F2=2 
A=5
t=np.arange(0,1,0.001)
x=A*np.sin(2*np.pi*F1*t)
u=[]
b=[0.2,0.4,0.6,0.8,1.0]

s=1
for i in t:
    if(i==b[0]):
        b.pop(0)
        if(s==0):
            s=1
        else:
            s=0
       
    u.append(s)
v=[]
for i in range(len(t)):
    v.append(A*np.sin(2*np.pi*F1*t[i])*u[i])
    
plt.subplot(3,3,1)
plt.plot(t,x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Carrier')
plt.grid(True)

plt.subplot(3,3,4)
plt.plot(t,u)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(3,3,7)
plt.plot(t,v)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('ASK Signal')
plt.grid(True)
plt.tight_layout()


# FREQUENCY SHIFT KEYING............................

t=np.arange(0,1,0.001)
x=A*np.sin(2*np.pi*F1*t)
 
plt.subplot(3,3,2)
plt.plot(t,x)
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Carrier")
plt.grid(True)

u=[]
b=[0.2,0.4,0.6,0.8,1.0]
s=1
for i in t:
    if(i==b[0]):
        b.pop(0)
        if(s==0):
            s=1
        else:
            s=0
    u.append(s)

plt.subplot(3,3,5)
plt.plot(t,u)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Message Signal')
plt.grid(True)

v=[]
for i in range(len(t)):
    if(u[i]==1):
        v.append(A*np.sin(2*np.pi*F1*t[i]))
    else:
        v.append(np.sin(2*np.pi*F1*t[i])*-1)

plt.subplot(3,3,8)
plt.plot(t,v)
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("FSK")
plt.grid(True)
plt.tight_layout()



#PHASE SHIFT KEYING................................

t=np.arange(0,1,0.001)
x=A*np.sin(2*np.pi*F1*t)

plt.subplot(3,3,3)
plt.plot(t,x)
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Carrier")
plt.grid(True)

u=[]
b=[0.2,0.4,0.6,0.8,1.0]
s=1
for i in t: 
    if(i==b[0]):
        b.pop(0)
        if(s==0):
            s=1
        else:
            s=0
    u.append(s)

plt.subplot(3,3,6)
plt.plot(t,u)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Message Signal')
plt.grid(True)

v=[]
for i in range(len(t)):
    if(u[i]==1):
        v.append(A*np.sin(2*np.pi*F1*t[i]))
    else:
        v.append(A*np.sin(2*np.pi*F1*t[i])*-1)

plt.subplot(3,3,9)
plt.plot(t,v)
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("PSK")
plt.grid(True)
plt.tight_layout()
plt.show()
# A to A...................................................................

#Amplitude Modulation.............................
A_c = 3 #float(input('Enter carrier amplitude: '))
f_c = 40 #float(input('Enter carrier frquency: '))
A_m = 2 #float(input('Enter message amplitude: '))
f_m = 4 #float(input('Enter message frquency: '))


t = np.linspace(0, 1, 1000)

carrier = A_c*np.cos(2*np.pi*f_c*t)
modulator = A_m*np.cos(2*np.pi*f_m*t)
product = A_c*(1+np.cos(2*np.pi*f_m*t))*np.cos(2*np.pi*f_c*t)

plt.subplot(3,3,1)
plt.title('Amplitude Modulation')
plt.plot(modulator)
plt.ylabel('Amplitude')
plt.xlabel('Message signal')

plt.subplot(3,3,4)
plt.plot(carrier)
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')

plt.subplot(3,3,7)
plt.plot(product)
plt.ylabel('Amplitude')
plt.xlabel('AM signal')

plt.subplots_adjust(hspace=1)
# plt.show()


#Frequency Modulation..........................

modulator_frequency = 4
carrier_frequency = 40
modulation_index = 1

time = np.linspace(0,1,1000)
modulator = np.sin(2 * np.pi * modulator_frequency * time) * modulation_index
carrier = np.sin(2 * np.pi * carrier_frequency * time)
product = np.zeros_like(modulator)

for i, t in enumerate(time):
    product[i] = np.sin(2 * np.pi * (carrier_frequency * t + modulator[i]))

plt.subplot(3, 3, 2)
plt.title('Frequency Modulation')
plt.plot(time,modulator)
plt.ylabel('Amplitude')
plt.xlabel('Message signal')

plt.subplot(3, 3, 5)
plt.plot(time,carrier)
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')

plt.subplot(3, 3, 8)
plt.plot(time,product)
plt.ylabel('Amplitude')
plt.xlabel('FM signal')
plt.tight_layout()


#Phase Modulation.............................
carrier = 200.0
modulator = 800.0
beta = 1.0
x1 = np.linspace(0.0, 0.03, num=2000)
y1 = np.sin(carrier * np.pi * x1)
y2 = np.cos(modulator * np.pi * x1)
y3 = np.sin(carrier * np.pi * x1 + beta * y2)
plt.subplots_adjust(left = 0.1, bottom = 0.25)

plt.subplot(3, 3, 3)
plt.plot(x1, y1)
plt.title('Phase Modulation')
plt.ylabel('Amplitude')
plt.xlabel('Message Signal')

plt.subplot(3, 3, 6)
plt.plot(x1, y2)
plt.ylabel('Amplitude')
plt.xlabel('Carrier Signal')

plt.subplot(3, 3, 9)
plt.plot(x1, y3)
plt.ylabel('Amplitude')
plt.xlabel('PM Signal')
plt.tight_layout()
plt.show()