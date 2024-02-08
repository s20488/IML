''' 
Zapis/odczyt danych do/z plików, w tym YAML
'''

# Otwarcie pliku "dane.txt" do zapisu ("w")
# Odczyt: "r"; odczyt/zapis: "r+"; dołączanie danych na końcu pliku: "a"
plik = open("dane.txt", "w")
for i in range(10):
    plik.write(str(i) + "\n")  # Zapisujemy kolejne liczby w osobnych wierszach
    # Do pliku zapisujemy ciągi tekstowe, stąd funkcja str() konwertująca liczbę na tekst
    # Ciąg "\n" oznacza rozpoczęcie nowego wiersza
plik.close() # Zamknięcie pliku

#-------------

# Otwarcie pliku do odczytu ("r")
plik = open("dane.txt", "r")
# Wczytanie zawartosci całego pliku i wyświetlenie jej na ekranie
print("Cały plik:")
print(plik.read())

#-------------

# Jak wczytać zawartość pliku do listy?

plik.seek(0)  # "Przewijamy" plik na początek, bo wcześniej był już odczytany
lista = []    # Pusta lista
for i in range(10):
    lista.append(plik.readline())   # Wczytanie jednej linii z pliku  i dołączenie do listy
plik.close()

print("Wczytana lista:", lista)

# Wczytana lista ma elementy typu tekstowego, musimy wykonać konwersję np. na typ "int"
for i in range(len(lista)):
    lista[i] = int(lista[i])

# To samo bardziej po "pythonowemu":
lista = [int(i) for i in lista]

print("Lista po konwersji na liczby:", lista)


#-------------

# Zapis do pliku YAML

import yaml
import io

# Dane do zapisu
data = {'lista': [1, 42, 3.141, 'tekst', '€'],
        'napis': 'bla',
        'slownik': {'foo': 'bar', 'sens życia': 44}}
print(data)
# UWAGA! W konsoli Windows próba wydruku znaku w UTF-8 wyrzuci błąd
# "UnicodeEncodeError: 'charmap' codec can't encode character..."
# Aby go uniknąć należy przełączyć konsolę na UTF-8 poleceniem "chcp 65001"
# i ponownie wykonać program

# Zapis pliku YAML
with io.open('dane.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# Wczytanie pliku YAML
with open("dane.yaml", 'r', encoding='utf8') as stream:
    data_loaded = yaml.safe_load(stream) # safe_load dla prostych plików YAML
print(data_loaded)

print(data == data_loaded)

