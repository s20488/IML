'''
Podstawowe typy danych i operacje na nich
Funkcja print()
Funkcja input()
Podstawowe użycie instrukcji warunkowej i pętli
'''

# Liczby
liczba1 = 10
liczba2 = 3.1415
# Wielkość liter w nazwach zmiennych ma znaczenie

print(liczba1, type(liczba1))
print(liczba2, type(liczba2))

# Podstawowe operacje arytmetyczne
print(liczba1 + liczba2, liczba1-liczba2, liczba1*liczba2, liczba1/liczba2, liczba1**liczba2)


# Stringi (napisy)
napis = "To jest napis"
print(napis, type(napis))
# Cudzysłowy i apostrofy są równoważne

# Wyswietlanie fragmentu stringu
print(napis[0])
print(napis[8:])
print(napis[3:7])
print(napis[:3])

# Co druga litera
print(napis[::2])

# Co trzecia litera, począwszy od 2
print(napis[1::3])

# Chcemy zmienić w stringu pierwszą literę na "t"
# Stringi są niemutowalne, więc to nie zadziała: napis[0]="t"
# Trzeba zrobić tak:
napis = "t" + napis[1:]
print(napis)  # udało się 

# Kilka metod działających na napisach
print(napis.capitalize())  # Pierwsza litera -> duża
print(napis.count("a"))    # Policzenie liter "a"
print(napis.replace("jest", "był"))  # Zamiana fragmentu na inny

# "Pomnożenie" napisu przez 2
print(napis*2) 

# Sprawdzenie, czy napis można przekonwertować na liczbę
print(napis.isdigit())

# To samo z innym napisem, który da się przekonwertować:
innynapis = "100"
print(type(innynapis)) # to ciągle jest string
print(innynapis.isdigit())

# Wykorzystanie w praktyce - instrukcja warunkowa
if innynapis.isdigit():
    liczba = int(innynapis)
    print(liczba)
else:
    print("Tego napisu nie da się przekonwertować na liczbę")

# Typ logiczny
test = innynapis.isdigit()  # True lub False
print(test)
print(type(test))

if test:
    print("OK")

# Napis wielowierszowy
napis_wielowierszowy = '''To jest
napis
wielowierszowy!'''

print(napis_wielowierszowy)

# Podział na poszczególne linie
linie = napis_wielowierszowy.split('\n')
print(linie) # W wyniku otrzymujemy listę

# Pętla drukująca elementy listy
for linia in linie:
    print(linia)

# Wczytywanie danych z klawiatury
liczba = input("Podaj liczbę:")
print(liczba,type(liczba)) # Jak widać, wczytana "liczba" jest stringiem
# Przecinek w funkcji print() domyslnie dodaje spację
liczba = int(liczba)
print(liczba,type(liczba))

# Wczytanie kilku danych jednoczesnie
d,m,r = input("Podaj dzień, miesiąc, rok:").split(",") # Uwaga na separator!

# Zmiana domyslnego separatora i zakonczenia wydruku
print("Podano datę", end=": ")
print(d,m,r, sep='.', end=' r.\n') 

# To samo z wykorzystaniem formatowania stringu:
print("Podano datę: {}.{}.{} r.".format(d,m,r))

# Inny sposób (tzw. f-string):
print(f"Podano datę: {d}.{m}.{r} r.")

# Długość linii w Pythonie: 79 znaków (rekomendacja), ew. 100. 
# Nie ma jednak formalnego limitu, chodzi o czytelność kodu
# Automatyczna kontynuacja wiersza: przecinek lub nawias:
print("To jest",
    "kontynuacja")

print(
    "To też jest kontynuacja"
    )

# Lista (będzie omówiona później)
T = [
    1,2,3,
    4,5,6,
    7,8,9
    ]
print(T)

# Linie kontynuowane powinny być zapisywane z wcięciem (ale nie jest to wymaganie ścisłe)

# Wymuszona kontynuacja wiersza: znak "\":
print\
    ("To jest kontynuacja")
