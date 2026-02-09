import requests
import pandas as pd
import time

rows = []

for pokemon_id in range(1, 152):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    response = requests.get(url)

    if response.status_code != 200:
        continue

    data = response.json()

    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

    row = {
        "hp" : stats["hp"],
        "attack" : stats["attack"],
        "defense" : stats["defense"],
        "special_attack" : stats["special-attack"],
        "speed" : stats["speed"],
        "base_experience" : data["base_experience"]
    }

    rows.append(row)
    time.sleep(0.2)

df = pd.DataFrame(rows)
df.to_csv("pokemon_dataset.csv", index=False)

print("Saved:", df.shape)

#check data
df.info()
df.describe()
df.head()