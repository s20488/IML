import yaml
import os
import sys
from pick import pick  # python -m pip install pick

# Collect name and load YAML file to proceed with
input_filenames = []
for filename in os.listdir("."):
    if filename.endswith(".yaml"):
        input_filenames.append(filename)
input_filenames.sort()

if not input_filenames:
    sys.exit("Brak plików YAML w katalogu!")

filename, _ = pick(input_filenames, "Wybierz plik YAML:", indicator='>')

file = open(filename, "r", encoding='utf8')
data = yaml.safe_load(file)

# Select list of facts for calculations

# Collect hypotheses names in a list
h_names = []
for h in data["Hypotheses"]:
    h_names.append(h["name"])

# Collect facts names in a list
facts_names = []
for index, fact in enumerate(data["Facts"]):
    facts_names.append(fact["name"])

# Get selected facts as tuple ( (fact, index), (fact, index), ... )
selected_facts_tuple = pick(facts_names,
                            "Badane hipotezy: " + ", ".join(
                                h_names) + ".\nZaznacz fakty spacją i zatwierdź wciskając Enter:",
                            multiselect=True)

# Convert selected facts to a list [fact, fact, ...]
selected_facts = [i[0] for i in selected_facts_tuple]

# Print probabilities a prori
print("\nPRAWDOPODOBIEŃSTWA A PRIORI")
for h in data["Hypotheses"]:
    print("Prawdopodobieństwo hipotezy '{}' wynosi {:.2f}%".format(h["name"], h["prob"] * 100))

# Calculate probability of facts
print("\nPRAWDOPODOBIEŃSTWA WYSTĄPIENIA FAKTÓW")
Pr_f = []
for fact in data["Facts"]:
    sum = 0
    for index, h in enumerate(data["Hypotheses"]):
        sum = sum + h["prob"] * fact["prob"][index]
    Pr_f.append([fact["name"], sum])

for pr in Pr_f:
    print("Prawdopodobieństwo wystąpienia faktu '{}' wynosi {:.2f}%".format(pr[0], pr[1] * 100))

# Calculate probability of hypothesis under a single fact
Pr_h_f = []
for indexh, h in enumerate(data["Hypotheses"]):
    for indexf, fact in enumerate(data["Facts"]):
        pr = h["prob"] * fact["prob"][indexh] / Pr_f[indexf][1]
        Pr_h_f.append([h["name"], fact["name"], pr])

print("\nPRAWDOPODOBIEŃSTWA A POSTERIORI, UWZGLĘDNIAJĄCE POJEDYNCZE FAKTY")
for pr in Pr_h_f:
    print("Prawdopodobieństwo hipotezy '{}' przy uwzględnieniu faktu '{}' wynosi: {:.2f}%" \
          .format(pr[0], pr[1], pr[2] * 100))

# Calculate probability of hypothesis under a list of facts
print("\nPRAWDOPODOBIEŃSTWA A POSTERIORI, UWZGLĘDNIAJĄCE KILKA FAKTÓW JEDNOCZEŚNIE")
print("Wybrane fakty: '{}' ".format(", ".join(selected_facts)))

'''
Zadanie:
a) Uzupełnić program tak, aby uwzględniał dowolne zestawy faktów
   wg. wzoru w żółtej obwódce w pliku PDF.
b) Utworzyć nowy plik z definicjami faktów, który mógłby posłużyć do wnioskowania 
   na temat dowolnie wybranego zestawu co najmniej 3 hipotez i co najmniej 3 faktów
   (np. COVID 19 / grypa / przeziębienie / inna choroba; gorączka / kaszel / utrata węchu i smaku / katar).
'''

denumerator = 0
for ih, h in enumerate(data["Hypotheses"]):
    # Calculate the denumerator of the formula
    value = 1  # Should be replaced by the actual calculations
    for index, fact in enumerate(data["Facts"]):
        if fact["name"] in selected_facts:
            value *= fact["prob"][ih]
    denumerator += h["prob"] * value

for ih, h in enumerate(data["Hypotheses"]):
    # Calculate the numerator of the formula
    value = 1  # Should be replaced by the actual calculations
    for index, fact in enumerate(data["Facts"]):
        if fact["name"] in selected_facts:
            value *= fact["prob"][ih]
    numerator = h["prob"] * value

    # Calculate and print the final result
    pr = numerator / denumerator
    print("Prawdopodobieństwo hipotezy '{}' przy uwzględnieniu powyższych faktów wynosi: {:.2f}%" \
          .format(h["name"], pr * 100))
