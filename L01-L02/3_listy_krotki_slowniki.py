'''
Listy, krotki, słowniki
a przy okazji pętle i instrukcje warunkowe
'''

# Lista jednowymiarowa
lista = [1,4,"python",3.1415]  
print(lista)
print()

# wypisz drugi element listy
print(lista[1])   # elementy numerujemy od zera - w wyniku dostaniemy '4'
print()

# wypisz elementy od drugiego do końca
print(lista[1:])
print()

# Wypisz elementy od drugiego do trzeciego
# Należy pamiętać, że zakres jest przedziałem prawostronnie otwartym,
# dlatego podajemy górną wartość o 1 większą  
print(lista[1:3])
print()

# Wypisz co drugi element
print(lista[::2])
print()

# Wypisz co trzeci element, począwszy od drugiego
print(lista[1::3])
print()

# Wypisz wszystkie elementy listy jeden pod drugim
for element in lista:
    print(element)
print()

# Wypisz wszystkie elementy listy jeden pod drugim w formacie:
# element nr [numer] = [wartosc]
for i in range(len(lista)):
    print("element nr", i, "=", lista[i])
print()

# Dodanie nowego elementu do listy
lista.append("nowy element")
print(lista)
print()

# Usunięcie elementu z listy po wartości - usuwamy wartość 3.1415
lista.remove(3.1415)
print(lista)
print()

# Usunięcie elementu z listy po indeksie - usuwamy pierwszy element
del lista[0]
print(lista)
print()

# Zmieniamy pierwszy element listy na wartość 10
lista[0] = 10
print(lista)
print()

# długosć listy
print(len(lista))
print()

# Sprawdzenie, czy dana wartość występuje na liście
szukana_wartosc = 10
# Wykorzystujemy instrukcję warunkową
if szukana_wartosc in lista:
    print("Znaleziono", szukana_wartosc, "na liscie")
else:
    print("Nie znaleziono", szukana_wartosc, "na liscie")
print()


# Pętla z wykorzystaniem indeksów elementów listy
for indeks,wartosc in enumerate(lista):
    print(indeks, wartosc, sep=": ")

print()

#---------------------------------------------------

# Lista dwuwymiarowa
lista2= [[0,1,2],[3,4,5],[6,7,8]]
print(lista2[1][2])  # trzeci element drugiego wiersza
print()

# Wypisanie wszystkich elementów
for wiersz in lista2:               # w wyniku dostajemy kolejny wiersz 
    for element_wiersza in wiersz:  # teraz wybieramy kolejne elementy z wiersza
         print(element_wiersza)     # i wypisujemy je na ekranie
print()


#---------------------------------------------------

# Generowanie list

# Lista 0,1,...,9
lista = list(range(0,10))
print(lista)
print()

# Lista 5,7,...,99
lista = list(range(5,100,2))
print(lista)
print()

# "Hurtowa" operacja na liscie - mnożenie przez 2
lista = [i*2 for i in lista]
print(lista)
print()  

#---------------------------------------------------

# Krotka (tuple)

# Przekształcenie listy w krotkę
krotka = tuple(lista)
print(krotka)
print()

# Sprawdzenie, ile krotka ma elementów i wypisanie odpowiedniego komunikatu
ile = len(krotka)
if ile == 0:
    print("Ta krotka jest pusta!")
elif 1 <= ile <= 5:
    print("Ale krótka krotka!")
else:
    print("Ta krotka ma więcej niż 5 elementów, a dokładnie jest ich", ile)
print()

# Krotka jest niemutowalna - nie można zmieniać jej elementów, 
# więc np.to nie zadziała: krotka[0] = 5   itp.

# Przekształcenie krotki w listę
lista = list(krotka)
print(lista)
print()


# Funkcja "zip" - łączenie list (ogólnie różnych iterabli) w krotkę
names = ["Maria", "Anna", "Jerzy"]
surnames = ["Nowak", "Kowalska", "Iksiński"]
people = zip(names, surnames)
print(list(people))

print()

#---------------------------------------------------

# Słownik (dictionary)

# Deklaracja słownika i przypisanie kilku par klucz->wartość
pracownik1={}
pracownik1["imie"] = "Janusz"
pracownik1["nazwisko"] = "Kowalski"
pracownik1["wiek"] = 30

pracownik2={}
pracownik2["imie"] = "Maria"
pracownik2["nazwisko"] = "Nowak"
pracownik2["wiek"] = 33

# Wyświetlanie danych "pojedyńczo" w jednej linii
print(pracownik1["imie"], pracownik1["nazwisko"], pracownik1["wiek"])
print(pracownik2["imie"], pracownik2["nazwisko"], pracownik2["wiek"])
print()

# Wyświetlenie danych w pętli
for klucz, wartosc in pracownik1.items():
    print(klucz, ":", wartosc)
print()

# Lista pracowników (zapisanych jako słowniki)
pracownicy = []
pracownicy.append(pracownik1)
pracownicy.append(pracownik2)

# Wyświetlenie całości
print(pracownicy)
print()

# Wyświetlenie wieku pierwszego pracownika
print(pracownicy[0]["wiek"])
print()

# Wyświetlenie wszystkich pracowników - w pętli
for pracownik in pracownicy:
    print(pracownik["imie"], pracownik["nazwisko"], pracownik["wiek"])
print()

# Słownik "dwuwymiarowy" (dodajemy stanowisko pracownika)
pracownicy = {"kadry": {"imie":"Jan", "nazwisko":"Kowalski", "wiek":30}, 
              "płace": {"imie":"Maria", "nazwisko":"Nowak", "wiek":33}}
# Albo tak, gdy już mamy poszczególnych pracowników jako słowniki:
pracownicy = {"kadry": pracownik1, "płace": pracownik2}
# Uwaga! w słowniku nie mogą się powtarzać klucze, więc w ten sposób nie 
# da się przypisać kilku pracowników do jednego stanowiska.
# Można w zamian umieścić pracowników na danym stanowisku na liście, np.:
# pracownicy = {"kadry": [pracownik1, pracownik2], "płace": [pracownik3]}

# Wyświetlenie wieku pracownika kadr
print(pracownicy["kadry"]["wiek"])
print()

# Postarzenie pracownika kadr o 1 rok
pracownicy["kadry"]["wiek"] += 1
print(pracownicy["kadry"]["wiek"])
print()

# Dodanie kolejnego pracownika
pracownik3 = {}
pracownik3["imie"] = "Józef"
pracownik3["nazwisko"] = "Kowalski"
pracownik3["wiek"] = 44
pracownicy.update({"księgowość":pracownik3})

# Wyświetlenie w pętli wszystkich pracowników wraz z ich stanowiskami
for stanowisko, dane in sorted(pracownicy.items()):
    print(stanowisko,":",dane["imie"], dane["nazwisko"], dane["wiek"])
print()


# Wyświetlenie w pętli wszystkich pracowników wraz z ich stanowiskami,
# posortowanych malejąco wg. wieku: stanowisko: imię nazwisko, wiek "lat"
for stanowisko, dane in sorted(pracownicy.items(), reverse=True, key=lambda x: x[1]['wiek']):
    if str(dane["wiek"]).endswith("2") or str(dane["wiek"]).endswith("3") or str(dane["wiek"]).endswith("4"):
        str_wiek = "lata"
    else:
        str_wiek = "lat"    
    print("{:>10}: {} {}, {} {}".format(stanowisko, dane["imie"], dane["nazwisko"], dane["wiek"], str_wiek))
print()

