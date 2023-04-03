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


def delta_train(training_data, learning_rate, epochs):
    num_attributes = len(training_data[0][0])
    weights = [random.uniform(-0.5, 0.5) for _ in range(num_attributes)]

    for _ in range(epochs):
        for attributes, decision in training_data:
            output = sum([attributes[i] * weights[i]
                         for i in range(num_attributes)]) > 0
            target = decision == 'Iris-setosa'
            for i in range(num_attributes):
                weights[i] += learning_rate * (target - output) * attributes[i]

    return weights


def predict(weights, attributes):
    return 'Iris-setosa' if sum([attributes[i] * weights[i] for i in range(len(weights))]) > 0 else 'Iris-versicolor_or_virginica'
