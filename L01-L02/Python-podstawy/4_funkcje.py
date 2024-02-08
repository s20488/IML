'''
Funkcje
a przy okazji wczytywanie danych z klawiatury, instrukcja warunkowa
'''

# Definiujemy funkcję, podnoszącą argument do kwadratu
# Pamiętajmy, że używane w funkcjach zmienne mają zasięg lokalny
def kwadrat(x):
    return x**2  # Zwrócenie wartosci

# To jest "program główny"
liczba = input("Podaj liczbe: ")  # "input" zawsze wczytuje string
liczba = float(liczba)  # konwersja na typ zmiennoprzecinkowy
wynik = kwadrat(liczba)  # wywołanie funkcji
print("liczba =", liczba)
print("kwadrat =", wynik) # wyświetlenie wyniku


# Funkcja jako procedura, nie zwracająca wartości
def wypisz(x):
    print("Podano liczbę", x)
    if  10 < x < 20:
        print("Jest to liczba z przedzialu (10, 20)")
    else:
        print("Jest to liczba spoza przedzialu (10, 20)")


# Dalsza część "programu głównego"
liczba = input("Podaj drugą liczbę: ")
wypisz(int(liczba))    # wywołanie funkcji w trybie "procedury", z konwersją typu
