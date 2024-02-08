'''
Różne rodzaje pętli
Instrukcja warunkowa
'''


# Pętla wyświetlająca liczby od 0 do 9
for i in range(10):
    print(i)            # Instrukcja w pętli (wcięcie!)

print()                 # Instrukcja poza pętlą (brak wcięcia)

# Pętla od i=0 do i=9 z krokiem 1, wyświetlająca liczbę i, jej kwadrat i sześcian
for i in range(0,10):
    print(i, i**2, i**3),

print()         

# To też jest pętla od 0 do 9
i = 0
while i < 10:
    print(i)
    i = i+1

print()

# Pętla while z wyłapaniem wyjątku
while True:
    try:
        x = int(input("Wpisz liczbę: "))
        break # Jeżeli podano liczbę, to pętla się przerwie
    except ValueError:
        print("Miała być liczba, spróbuj ponownie!")
        
print("Podano liczbę {}".format(x))


print()

# Pętla od 5 do 9 z krokiem 2
for i in range(5, 10, 2):
    print(i)

print()

# Generalnie pętla "for" iteruje po liście, krotce, słowniku
lista = [1, "abc", 3.14]
for element in lista:
    print(element)
    
print()    
    
kolory = ("red", "green", "blue")
for kolor in kolory:
    print(kolor)

for kolor in kolory:
    # Instrukcja warunkowa
    if kolor == "red":
        print("czerwony")
    elif kolor == "green":
        print ("zielony")
    elif kolor == "blue":
        print ("niebieski")
    else:
        pass # instrukcja "pusta" - "nic nie rób"

print()    

x = int(input("Podaj liczbę: ")) 

if 0 < x < 10:
    print("ale mała liczba!")
elif x < 0:
    print("ujemna")
else:
    print("większa od 10!")
    

# Od Pythona w wersji 3.10:
match x:
    case x if 0 < x < 10:
        print("ale mała liczba!")
    case x if x < 0:
        print("ujemna")
    case _:
        print("większa od 10!")

# Prostsze i bardziej typowe wykorzystanie match... case
match x:
    case 0:
        print("Wpisano liczbę zero.")
    case 10:
        print("Wpisano liczbę dziesięć.")
    case _:
        print("Wpisano liczbę różną od zera i dziesięciu.")

