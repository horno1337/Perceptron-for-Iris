import random

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
    return 'Iris-setosa' if sum([attributes[i] * weights[i] for i in range(len(weights))]) > 0 else 'Inne'


def test_perceptron(test_data, weights):
    correct = 0
    for attributes, decision in test_data:
        prediction = predict(weights, attributes)
        if prediction == 'Iris-setosa' and decision == 'Iris-setosa':
            correct += 1
        elif prediction == 'Inne' and decision != 'Iris-setosa':
            correct += 1

    accuracy = (correct / len(test_data)) * 100
    return correct, accuracy


# Train the perceptron
learning_rate = 0.1
epochs = 100
weights = delta_train(train_data, learning_rate, epochs)

# Test the perceptron
correct, accuracy = test_perceptron(test_data, weights)
print(f'ilosc poprawnie sklasyfikowanych przykladow: {correct}')
print(f'dokladnosc: {accuracy:.2f}%')

# Manual input
while True:
    try:
        input_str = input(
            'Wpisz wartosci (po przecinku) lub \'q\' aby wyjsc: ')
        if input_str.lower() == 'q':
            break

        input_attributes = [float(x.strip()) for x in input_str.split(',')]
        print(f'Wynik: {predict(weights, input_attributes)}')
    except Exception as e:
        print('Error:', e)
