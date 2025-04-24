# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:15:18 2022

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
#import clifford as cf
from clifford.g3 import *
import imageio

# Los porductos y la base están incluidas en la libreria, con los simbolos de 2D
# El pseudoescalar es e123

# El producto exterior se tiene que hacer con parentesis según como esta hecha la libreria y python
# (e1^e2)+(e2^e3)

# El producto interior de un escalar y un multivector es cero en esta libreria
# Para escalares, usar el geometrico: 2*e1

# Los multivectores se construyen igual que en la teoria

# La reversión se efectúa con el símbolo ~. ~A == A^{\dagger}

# Se puede proyectar sobre un grado n de un multivector con paréntesis:
# El grado 0 de A es A(0), el elemento de grado 1 es A(1), etc.

# Se puede definir el módulo al cuadrado de un multivector así: (A*~A)(0) o abs(A)**2

# La inversa de un multivector se define A.inv(), y tenemos A.inv()*A = 1

# Podemos ver el dual de un elemento del algebra con a.dual(), el dual de un vector es un bivector (obviamente)

# Puedes decidir el tipo de display a la hora de imprimir los resultados:
# Con cf.ugly() al inicio, todo se muestra feo, pero con la info muy informático
# COn cf.pretty(), se muestra bonito, con sintaxis parecidad a la de teoría
# Se puede fijar una precisión de decimales con cf.pretty(precision=n), para n decimales

# =============================================================================
# Aplicaciones 
# =============================================================================

'''
Es literalmente ir a la teoría desarrollada en el TFG de álgebra geométrica,
definir los vectores, rotores y multivectores correspondientes, y usar las
expresiones obtenidas con los productos que toquen, y ya tienes todas las
operaciones.
'''

'''
# Reflexiones

a = e1+e2+e3    # a vector
n = e1          # the reflector normalized
b = n*a*n           # reflect `a` in hyperplane normal to `n`

# Si queremos hacer una reflexion sobre un vector no normalizado, tenemos que 
# usar la expresión b = n*a*n.inv()

# Si queremos la reflexión en el hiperplano normal a n, tenemos que ponerle 
# un menos a la ecuación de reflexion, como vimos en teoria. 
'''

'''
# Rotaciones

# Hacemos una rotación
a = e1 + 2*e2 + 3*e3 # Vector
R = np.exp(np.pi/4*e12) # Rotor
b = R*a*~R # La rotación de 90 grados antihoraria
'''

# =============================================================================
# Angulos de Euler
# =============================================================================

# Definimos una función para producir un rotor con los angulos de euler.
'''
def R_euler(phi,theta,psi):
    Rphi = np.exp(-phi/2*e12)
    Rtheta = np.exp(-theta/2*e23)
    Rpsi = np.exp(-psi/2*e12)
    
    return Rphi*Rtheta*Rpsi


# Ángulos de Euler
phi = np.pi/4
theta = np.pi/4
psi = np.pi/4
R = R_euler(phi,theta,psi) # Rotor de Euler
print(R)
'''

# =============================================================================
# Condiciones iniciales y parámetros
# =============================================================================
# Si A es un multivector, con (A).value o A.value, podemos obtener un array  
# con los valores expresados en la base geometrica correspondiente, es decir:
# si A = 1*e1 + 2*e2 + 1e123, esto es el siguiente array:
#      = np.array([0,1,2,0,0,0,0,1]), que tiene sentido

i1,i3 = 0.5,1 # Momentos principales de inercia
L = 1*e12 + 0.1*e23 # Momento angular total, cte
#L = 1*e12 # Momento angular tipico de un trompo, 
#con L coincidiendo en la direccion de Omega_r
w3 = 1 #rad/s , Velocidad angular en la componente/direccion e3 (cte), 
#si aumenta, aumenta la velocidad de giro.

R0 = 1 # Orientacion inicial del sólido a t=0
Omega_l = L/i1 # Expresion de Omega_l cte en funcion de L
Omega_r = w3*(i1-i3)/i1*e123*e3 # Expresion de Omega_r constante en e12

# EJE I1, en la direccion e1
I1_0 = i1*e1 # Eje principal i1(t), como se va moviendo
# EJE I2, en la direccion e1
I2_0 = i1*e2 # Eje principal i1(t), como se va moviendo
# EJE I3, en la direccion e1
I3_0 = i3*e3 # Eje principal i1(t), como se va moviendo

t = np.linspace(0,19,1000) # Intervalo de tiempo

O = np.array([0,0,0]) # Origen

# =============================================================================
# Solución de la ecuación del rotor analítica
# =============================================================================
def R(t):
    R_l = np.exp(-1/2*Omega_l*t)
    R_r = np.exp(-1/2*Omega_r*t)
    return R_l*R0*R_r

'''
Una vez tenemos una expresion de un rotor que depende del tiempo, para el 
problema del trompo simetrico que estamos resolviendo, podemos saber como va a 
evolucionar cualquier punto que peretenezca al sólido. Si yo diera una malla
de puntos incial, solo tendria que aplicarle el rotor a cada punto de la malla
y obtendria su evolucion temporal, y por ende la del solido. Para caracterizar  
el movimiento y hacernos una idea de cómo esta girando podemos estudiar la 
evolucion de los tres ejes principales de inercia I1, I2 e I3.
'''

# Podemos obtener los ejes principales de inercia con el tiempo, 
# guardando las componentes x,y,z

# La componente x del eje I1 que evoluciona con el tiempo
I1_fx = np.zeros(len(t)) 
# La componente y del eje I1 que evoluciona con el tiempo
I1_fy = np.zeros(len(t)) 
# La componente z del eje I1 que evoluciona con el tiempo
I1_fz = np.zeros(len(t)) 
# Matriz tridimensional con las componentes de x,y,z de I1
I1 = np.zeros((len(t),len(t),len(t))) 

I2_fx = np.zeros(len(t))
I2_fy = np.zeros(len(t)) 
I2_fz = np.zeros(len(t)) 
I2 = np.zeros((len(t),len(t),len(t))) 

I3_fx = np.zeros(len(t)) 
I3_fy = np.zeros(len(t)) 
I3_fz = np.zeros(len(t)) 
I3 = np.zeros((len(t),len(t),len(t))) 

T = np.zeros(len(t))

for i in range(0,len(t)):
    I1_f = R(t[i])*I1_0*~R(t[i]) # Evolucion para cada instante de tiempo,
    #aplicando los rotores a I1
    I1_fx[i] = I1_f.value[1]
    I1_fy[i] = I1_f.value[2]
    I1_fz[i] = I1_f.value[3]
    I1_t = np.array([I1_fx[i],I1_fy[i],I1_fz[i]])
    
    I2_f = R(t[i])*I2_0*~R(t[i]) # Evolucion para cada instante de tiempo,
    #aplicando los rotores a I2
    I2_fx[i] = I2_f.value[1]
    I2_fy[i] = I2_f.value[2]
    I2_fz[i] = I2_f.value[3]
    I2_t = np.array([I2_fx[i],I2_fy[i],I2_fz[i]])
    
    I3_f = R(t[i])*I3_0*~R(t[i]) # Evolucion para cada instante de tiempo, 
    #aplicando los rotores a I3
    I3_fx[i] = I3_f.value[1]
    I3_fy[i] = I3_f.value[2]
    I3_fz[i] = I3_f.value[3]
    I3_t = np.array([I3_fx[i],I3_fy[i],I3_fz[i]])
    
    #Energia cinetica
    T[i] = 1/2*(i3*w3**2)
    

# =============================================================================
# Gráficas    
# =============================================================================

plt.figure()
plt.title('Principal axe I1, x component')
plt.plot(t,I1_fx,c='red',label='$I1_{x}$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$I1_{x}$ ($kg\\cdot m^{2}$)')
plt.grid()

plt.figure()
plt.title('Principal axe I1, y component')
plt.plot(t,I1_fy,c='purple',label='$I1_{y}$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$I1_{y}$ ($kg\\cdot m^{2}$)')
plt.grid()

plt.figure()
plt.title('Principal axe I1, z component')
plt.plot(t,I1_fz,c='orange',label='$I1_{z}$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$I1_{z}$ ($kg\\cdot m^{2}$)')
plt.grid()

plt.figure()
plt.title('Kinetic energy')
plt.plot(t,T,c='blue',label='$T(s)$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$T (J)$')
plt.grid()

plt.figure()
plt.title('Principal axe I3, x component')
plt.plot(t,I3_fx,c='red',label='$I1_{x}$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$I1_{x}$ ($kg\\cdot m^{2}$)')
plt.grid()

plt.figure()
plt.title('Principal axe I3, y component')
plt.plot(t,I3_fy,c='purple',label='$I1_{y}$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$I1_{y}$ ($kg\\cdot m^{2}$)')
plt.grid()

plt.figure()
plt.title('Principal axe I3, z component')
plt.plot(t,I3_fz,c='orange',label='$I1_{z}$')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('$I1_{z}$ ($kg\\cdot m^{2}$)')
plt.grid()


# =============================================================================
# Animación
# =============================================================================
'''
fig = plt.figure()
ax = plt.axes(projection='3d')
simulacion = []

def animate(i):
    ax.clear()
    ax.set_xlim([min(I3_fx)-1,max(I3_fx)+1]) # Puede que vaya fuera
    ax.set_ylim([min(I3_fy)-1,max(I3_fy)+1])
    ax.set_zlim([min(I3_fz)-1,max(I3_fz)+1])
    ax.set_xlabel('$I_{x}$')
    ax.set_ylabel('$I_{y}$')
    ax.set_zlabel('$I_{z}$')
    ax.quiver(O[0],O[1],O[2],I1_fx[i],I1_fy[i],I1_fz[i], color='red')
    ax.quiver(O[0],O[1],O[2],I2_fx[i],I2_fy[i],I2_fz[i], color='green')
    ax.quiver(O[0],O[1],O[2],I3_fx[i],I3_fy[i],I3_fz[i], color='blue')
    #plt.savefig('sim.png')
    #simulacion.append(imageio.imread('sim.png'))


ani = FuncAnimation(fig, animate, frames=len(t), interval=1, repeat=False)
#imageio.mimsave('simulacion.gif',simulacion, duration=1)
#ani.save("Trompo1b.gif",writer='imagemagick',fps=60)
plt.show()
'''

'''
# Otra posibilidad
I1[0,0,0] = I1_0.value[1] # Condicion inicial si se hace de golpe
for i in range(1,len(t)):
    # Evolución para cada instante de tiempo, aplicando los rotores
    I1_f = R(t[i])*I1_0*~R(t[i]) 
    I1[i,0,0] = I1_f.value[1]
    I1[0,i,0] = I1_f.value[2]
    I1[0,0,i] = I1_f.value[3]
    
   
plt.figure()
plt.title('Componente x del eje principal I1')
plt.plot(t,I1[:,0,0],c='red',label='$I1_{x}$')
plt.legend(loc='best')
plt.xlabel('Tiempo (s)')
plt.ylabel('$I1_{x}$')
plt.grid()

plt.figure()
plt.title('Componente y del eje principal I1')
plt.plot(t,I1[0,:,0],c='purple',label='$I1_{y}$')
plt.legend(loc='best')
plt.xlabel('Tiempo (s)')
plt.ylabel('$I1_{y}$')
plt.grid()

plt.figure()
plt.title('Componente z del eje principal I1')
plt.plot(t,I1[0,0,:],c='orange',label='$I1_{z}$')
plt.legend(loc='best')
plt.xlabel('Tiempo (s)')
plt.ylabel('$I1_{z}$')
plt.grid()
'''

