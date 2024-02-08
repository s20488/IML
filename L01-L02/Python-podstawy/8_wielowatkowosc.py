'''
Wielowątkowość
'''

from threading import Thread
from random import randint
import time
 
# Dziedziczymy po Thread 
class Watek(Thread):
 
    def __init__(self):
        ''' Konstruktor. Musimy w nim zawołać konstruktor klasy Thread. '''
        Thread.__init__(self)
 
    def run(self):
        ''' Funkcja uruchamiana automatycznie, wywołaniem .start() na obiekcie '''
        for i in range(1, 4):
            print('{}: i = {}'.format(self.name, i))
 
            # Pauza o losowej długości
            secondsToSleep = randint(5, 9)
            print('{}: czeka {} sekund...'.format(self.name, secondsToSleep))
            time.sleep(secondsToSleep)
        print('{}: skończył.'.format(self.name))
 
 
if __name__ == '__main__':
    
    # Deklaracja obiektów
    myThread1 = Watek()
    myThread1.name = 'Wątek 1'
 
    myThread2 = Watek()
    myThread2.name = 'Wątek 2'
 
    # Uruchomienie wątków
    myThread1.start()
    myThread2.start()
 
    # Oczekiwanie na zakończenie wątków
    myThread1.join()
    myThread2.join()
 
    print('Koniec...')