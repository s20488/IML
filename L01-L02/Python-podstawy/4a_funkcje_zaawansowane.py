'''
Funkcja lambda, lista argumentów, funkcja z argumentem opcjonalnym.
Wykorzystanie zmiennej __main__, opisów funkcji i funkcji help()
'''

def potega(x, n=1):
    '''
    Definiujemy funkcję, podnoszacą argument do wybranej potęgi, podawanej jako argument opcjonalny.
    Domyślnie jest to pierwsza potęga.
    '''    
    return x**n  # Zwrócenie wartosci

def potega1(x=1, n=2):
    '''
    Definiujemy funkcję, podnoszacą argument do wybranej potęgi.
    Oba argumenty są opcjonalne z domyślnymi wartościami.
    '''
    return x,n,x**n  # Zwrócenie wartosci w postaci krotki

def potega2(x=None, n=None):
    '''
    Definiujemy funkcję, podnoszacą argument do wybranej potęgi.
    Oba argumenty są opcjonalne, ale gdy nie zostaną podane, to funkcja odmawia obliczeń.
    '''
    if x and n:  # x i n nie mogą być "None" ani puste
        return x**n
    else:
        return "Podaj oba argumenty!"

def potega3(x=1, *lista_poteg):
    '''
    Pobieramy listę argumentów będących potęgami.
    Wynik od razu wypisujemy.
    '''
    for potega in lista_poteg:
        print("{}**{}={}".format(x,potega,x**potega))

def potega4(x=1, *lista_poteg):
    '''
    Pobieramy listę argumentów będących potęgami.
    Wynik zwracamy jako listę.
    '''
    wynik = []
    for potega in lista_poteg:
        wynik.append([x,potega,x**potega])
    return wynik
        

if __name__ == "__main__":
    # To jest "program główny".
    # Zmienna __name__ przechowuje wykonywany moduł, w naszym przypadku "__main__",
    # bo jest to moduł "główny", nie importowany z innego pliku

    # Funkcja lambda - mała, anonimowa funkcja
    suma = lambda a, b, c : a + b + c
    print(suma(5, 6, 12)) 

    # *args - lista argumentów
    suma = lambda *args: sum(args)
    print(suma(5, 6, 12, 8, 1)) 

    # Podobnie **kwargs - lista argumentów słownikowych
    print_prac = lambda **kwargs: print(kwargs)
    print_prac(p1="a", p2="b", p3="c")

    help(potega)

    liczba = input("Podaj liczbe: ")  # "input" zawsze wczytuje string
    liczba = float(liczba)  # konwersja na typ zmiennoprzecinkowy

    wynik1 = potega(liczba)  # wywołanie funkcji z domyślnym argumentem
    wynik2 = potega(liczba, 3)  # wywołanie funkcji z podaną wartością argumentu opcjonalnego
    wynik3 = potega1(n=3, x=liczba)  # parametry opcjonalne przekazane przez nazwę - w dowolnej kolejności
    wynik4 = potega1()  # parametry opcjonalne - oba domyślne

    print("liczba =", liczba)
    print("potega (n domyslne) =", wynik1)
    print("potega (n = 3) =", wynik2) 
    print("potega1 =", wynik3)
    print("potega1 =", wynik4)
        
    print(potega2())
    print(potega2(1))
    print(potega2(1,1))

    potega3(2,3,4,5) # Pierwszy argument to liczba, która będzie potęgowana, kolejne to potęgi
    print(potega4(2,3,4,5))
    