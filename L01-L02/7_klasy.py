''' 
Klasy, obiekty
'''

class osoba:
    '''Przykładowa klasa'''
    
    def __init__(self, imie, nazwisko, wiek):
        '''Konstruktor - wywołuje się automatycznie przy tworzeniu obiektu '''
        self.imie = imie
        self.nazwisko = nazwisko
        self.wiek = wiek
        # Uwaga! W Pythonie nie ma zmiennych prywatnych. Pewnym obejściem tego jest:
        # a) poprzedzenie zmiennej znakiem "_", np. "self._zmienna" co blokuje jej import 
        #    z innych modułów, lub
        # b) poprzedzenie zmiennej znakami "__", np. self.__zmienna, co powoduje, że dostęp 
        #    do zmiennej jest w postaci: "self._NazwaKlasy__zmienna".
        # Podobnie jest z metodami.
        # Nie należy tworzyć nazw w postaci "__nazwa__" - mają one specjalne znaczenie w Pythonie

    def postarz(self, o_ile=1):
        '''Przykładowa metoda'''
        self.wiek += o_ile
        
    def wypisz(self):
        from datetime import datetime
        rok_biezacy = datetime.now().year
        print("{} {} urodził się w {} roku.".format(self.imie, self.nazwisko, rok_biezacy-self.wiek))
        
    def __del__(self):
        '''Destruktor - wywołuje się automatycznie przy usuwaniu obiektu  '''
        pass  # Instrukcja "nic nie rób"
        
        
o1 = osoba("Jan", "Kowalski", 20)
o1.wypisz()

# Funkcja anonimowa, wypisująca atrybuty osoby
print_os = lambda o: print(o.imie, o.nazwisko, o.wiek)

# Testujemy "postarzanie" osoby
print_os(o1)
o1.postarz(10) # pomijamy "self"
print_os(o1)
o1.postarz() # bez argumentu - zostanie użyty domyślny
print_os(o1)


class pracownik(osoba):
    ''' Przykład dziedziczenia. 
    Można dziedziczyć z wielu klas - nazwy klas wymieniamy po przecinku.'''
    
    def __init__(self, imie, nazwisko, wiek, pensja, stanowisko):
        osoba.__init__(self, imie, nazwisko, wiek) # Tutaj nie pomijamy "self"
        self.pensja = pensja
        self.stanowiska = [stanowisko] # lista
        
    def podwyzka(self, kwota=0):
        self.pensja+= kwota
        print("Pracownik {} {} dostał podwyżkę w wysokości {} i zarabia teraz {}"\
        .format(self.imie, self.nazwisko, kwota, self.pensja))

    def awans(self, stanowisko, podwyzka=0):
        self.stanowiska.append(stanowisko) # Dopisanie do listy
        print("Pracownik {} {} awansował na stanowisko {}".format(self.imie, self.nazwisko, stanowisko))
        self.podwyzka(podwyzka)

    def wypisz(self):
        print("{} {} zarabia {} PLN na stanowisku {}.".format(self.imie, self.nazwisko, self.pensja, \
              str(self.stanowiska[-1])))


p1 = pracownik("Jerzy", "Nowak", 20, 15000, "programista")    
print(p1.imie, p1.nazwisko, p1.wiek, p1.pensja, p1.stanowiska)
p1.postarz(1)
p1.podwyzka(100)
print(p1.imie, p1.nazwisko, p1.wiek, p1.pensja)
p1.wypisz()

p1.awans("kierownik projektu")
p1.wypisz()

p1.awans("prezes", 10000)
p1.wypisz()

# Referencja do obiektu
p2 = p1
p2.awans("tester", -15000)
p2.wypisz()
p1.wypisz()


# Zadanie
# - Napisać klasę "pracownicy", która będzie przechowywała pracowników (klasa jak wyżej), 
#   przekazanych w konstruktorze w formie listy
# - Napisać metodę "wypisz_pracownikow", która przyjmuje dwa argumenty: 
#   "stanowisko" - nazwa stanowiska do wyświetlenia (argument obowiązkowy)
#   "obecnie" - opcjonalny argument logiczny określający czy metoda powinna wypisać pracowników 
#   zatrudnionych obecnie na danym stanowisku (True - wartość domyślna), 
#   czy też  pracowników którzy kiedykolwiek pracowali na danym stanowisku (False)
#   Metoda powinna wyświetlić wynik w następującej formie (przykład):
#   a) po wywołaniu .wypisz_pracownikow("programista"):
#   "Pracownicy obecnie na stanowisku programista: Imię Nazwisko, Imię Nazwisko..."
#   a) po wywołaniu .wypisz_pracownikow("programista", obecnie=False):
#   "Pracownicy kiedykolwiek na stanowisku programista: Imię Nazwisko, Imię Nazwisko..."
# - Przetestować działanie klasy i metody na obiekcie złożonym z 3 przykładowych pracowników


class pracownicy:
    
    def __init__(self, lista_pracownikow):
        self.lista = lista_pracownikow

    def wypisz_pracownikow(self, stanowisko, obecnie=True):
        if (obecnie):
            print("Pracownicy obecnie na stanowisku {}: ".format(stanowisko), end="")
            for p in self.lista:
                if stanowisko == p.stanowiska[-1]:
                    print(p.imie, p.nazwisko, end=", ")
        else:
            print("Pracownicy kiedykolwiek na stanowisku {}: ".format(stanowisko), end="")
            for p in self.lista:
                if stanowisko in p.stanowiska:
                    print(p.imie, p.nazwisko, end=", ")

    def wypisz_pracownikow1(self, stanowisko, obecnie=True):
        ''' Inny sposób '''
        l = [] # Lista pracowników spełniających warunek
        for p in self.lista:
            if (obecnie):
                if stanowisko == p.stanowiska[-1]:
                    l.append(p.imie + " " + p.nazwisko)
            else:
                if stanowisko in  p.stanowiska:
                    l.append(p.imie + " " + p.nazwisko)                
        print("Pracownicy {} na stanowisku {}: ".format("obecnie" if obecnie else "kiedykolwiek", stanowisko), end="")
        print(", ".join(l)) # Wyswietla listę, oddzielając przecinkami jej elementy


# Dane do przetestowania nowej klasy
p1 = pracownik("Jerzy", "Nowak", 20, 15000, "programista")    
p1.awans("kierownik projektu")
p2 = pracownik("Andrzej", "Iksiński", 20, 15000, "programista")
p3 = pracownik("Waldemar", "Zielony", 20, 7000, "programista")

# Obiekt nowej klasy
lp = pracownicy([p1, p2, p3])

# Testy metod
lp.wypisz_pracownikow("programista")
print()

lp.wypisz_pracownikow("programista", obecnie=False)
print()

lp.wypisz_pracownikow1("programista")
lp.wypisz_pracownikow1("programista", obecnie=False)

