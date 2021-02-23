import random

if not input:
    beat = {'R': 'P', 'P': 'S', 'S': 'R'}
    fusion = {'RP': 'a', 'PS': 'b', 'SR': 'c', 'PR': 'd', 'SP': 'e', 'RS': 'f', 'RR': 'g', 'PP': 'h', 'SS': 'i'}
    limits = [6, 16, 40]
    moves = ["", "", ""]
    numPredictors = 6 * len(limits) * len(moves)
    predictors = list(range(numPredictors))
    predictorscore = list(range(numPredictors))
    opponentscore = list(range(numPredictors))
    for i in range(numPredictors):
        predictors[i] = random.choice(['R', 'P', 'S'])
        predictorscore[i] = random.random();
        opponentscore[i] = random.random();
    metapredictors = [random.choice(['R', 'P', 'S']), random.choice(['R', 'P', 'S'])]
    metascore = 20
    threat = [0, 0, 0]
    outcome = 0
    length = 0
    output = random.choice(['R', 'P', 'S'])
else:
    oldoutcome = outcome
    if (beat[input] == output):
        outcome = 1
    elif (input == beat[output]):
        outcome = -1
    else:
        outcome = 0
    threat[oldoutcome + 1] *= 0.96
    threat[oldoutcome + 1] -= outcome
    for i in range(numPredictors):
        predictorscore[i] *= 0.8
        predictorscore[i] += (beat[input] == predictors[i])
        predictorscore[i] -= (input == beat[predictors[i]])
        opponentscore[i] *= 0.8
        opponentscore[i] += (beat[output] == predictors[i])
        opponentscore[i] -= (output == beat[predictors[i]])
    metascore *= 0.96
    metascore += (beat[input] == metapredictors[0])
    metascore -= (beat[input] == metapredictors[1])
    metascore -= (input == beat[metapredictors[0]])
    metascore += (input == beat[metapredictors[1]])
    moves[0] += input
    moves[1] += output
    moves[2] += fusion[input + output]
    length += 1

    for z in range(3 * len(limits)):
        j = min([length - 1, limits[z // 3]])
        while not moves[z % 3][length - j:length] in moves[z % 3][0:length - 1]:
            j -= 1
        i = moves[z % 3].rfind(moves[z % 3][length - j:length], 0, length - 1)
        predictors[2 * z] = moves[0][j + i]
        predictors[2 * z + 1] = moves[1][j + i]
    for i in range(numPredictors // 3, numPredictors):
        predictors[i] = beat[predictors[i - numPredictors // 3]]
    metapredictors[0] = predictors[predictorscore.index(max(predictorscore))]
    metapredictors[1] = beat[predictors[opponentscore.index(max(opponentscore))]]

    if random.random() < 0.2 * threat[outcome + 1] + 0.1:
        output = random.choice(['R', 'P', 'S'])
    elif metascore >= 0:
        output = metapredictors[0]
    else:
        output = metapredictors[1]