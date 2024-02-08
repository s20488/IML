''' 
Wyjątki
'''

# Proszę spróbować z poniższą instrukcją zakomentowaną
test = 1
try:    
    print(test)    # Próba wykonania instrukcji
except NameError:
    print("Zmienna 'test' nie istnieje!")  # Obsłużenie wyjątku; można dodać kolejne
else:
    print("Wszystko OK, zmienna 'test' istnieje.")   # Instrukcje, gdy wszystko poszło dobrze (opcjonalnie)


while True:
    try:
        x = int(input("Wpisz liczbę: "))
        break # Jeżeli podano liczbę, to pętla się przerwie
    except ValueError:
        print("Miała być liczba, spróbuj ponownie!")
        
print("Podano liczbę {}".format(x))


# Wyrzucenie wyjątku - na przykład ValueError
if x > 100:
    raise ValueError('Za duża liczba!')
