'''
Moduły
'''

import math    # Importujemy moduł math w całości
x = float(input("Podaj x = "))
# Odwołanie się do funkcji/stałych z modułu: math.funkcja(argument); math.stała
print("sin(pi*x) =", math.sin(math.pi*x))
print("ln(x) =", math.log(x))

# ---------------------------------

# Alternatywnie, możemy zrobić tak:
from math import sin,pi,log  # Importujemy tylko potrzebne nam funkcje/stałe
# Możemy się teraz do nich odwoływać bezpośrednio: funkcja(argument); stała
print("sin(pi*x) =", sin(pi*x))
print("ln(x) =", log(x))

# ---------------------------------

# Doczytanie funkcji ze zmianą jej nazwy:
from math import factorial as silnia
print("20!=",silnia(20))

# ---------------------------------

# Import modułu NumPy - metody numeryczne i nie tylko 
import numpy as np   # Zwyczajowo importujemy jako "np"
mediana = np.median([1,2,3])
print(mediana)

# ---------------------------------

# Import modułu pyplot (wykresy) z większego modułu matplotlib
import matplotlib.pyplot as plt
x = np.arange(0, 2*pi, 0.01); # Jak "range", ale krok może być ułamkowy
y = np.sin(x)
plt.plot(x, y)  # Wykres sin(x) dla x[0..2*pi]

