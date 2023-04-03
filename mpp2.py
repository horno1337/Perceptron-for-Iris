# wczytaj dane z pliku
def wczytaj_dane(plik):
    with open(plik, 'r') as f:
        lines = f.readlines()

    # podziel na atrybuty i decyzje
    dane = []
    for line in lines:
        attributes = [float(x.replace(',', '.')) for x in line.split()[:-1]]
        decision = line.strip().split('\t')[-1].strip()
        dane.append((attributes, decision))

    return dane


# wczytaj dane
train_data = wczytaj_dane('iris_training.txt')
test_data = wczytaj_dane('iris_test.txt')
