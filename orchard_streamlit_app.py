import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# === SPECIES SETUP ===
SPECIES = [
    {"name": "Golden",     "id": 0, "susceptibility": "high"},
    {"name": "Liberty",    "id": 1, "susceptibility": "low"},
    {"name": "Enterprise", "id": 2, "susceptibility": "low"},
]

COLOR_MAP = {
    0: "#7FB77E",  # Golden
    1: "#F4A261",  # Liberty
    2: "#8D6E63",  # Enterprise
}

CLUSTER_PENALTY = 0.1

# === STREAMLIT UI ===
st.title("Apple Orchard Layout Simulator")
st.write("Design a multi-species orchard that limits disease spread.")

num_rows = st.slider("Number of Rows", 3, 20, 6)
num_cols = st.slider("Trees per Row", 10, 50, 25)

st.subheader("Species Probabilities")
probabilities = {}
for species in SPECIES:
    prob = st.slider(
        f"Probability of planting {species['name']}",
        0.0, 1.0, 0.3
    )
    probabilities[species["id"]] = prob

# Normalize probabilities
total = sum(probabilities.values())
if total == 0:
    st.error("Total probability cannot be zero.")
    st.stop()
norm_probs = {k: v / total for k, v in probabilities.items()}

def choose_species(i, j, grid):
    probs = []
    for s in SPECIES:
        prob = norm_probs[s["id"]]
        if s["susceptibility"] == "high":
            if j >= 2 and grid[i, j-1] == s["id"] and grid[i, j-2] == s["id"]:
                prob *= CLUSTER_PENALTY
            if i >= 2 and grid[i-1, j] == s["id"] and grid[i-2, j] == s["id"]:
                prob *= CLUSTER_PENALTY
        probs.append(prob)
    total = sum(probs)
    probs = [p / total for p in probs]
    return np.random.choice([s["id"] for s in SPECIES], p=probs)

def generate_orchard():
    grid = np.full((num_rows, num_cols), -1)
    for i in range(num_rows):
        for j in range(num_cols):
            grid[i, j] = choose_species(i, j, grid)
    return grid

if st.button("Simulate Orchard"):
    grid = generate_orchard()
    fig, ax = plt.subplots(figsize=(12, 2.5))
    color_array = np.vectorize(COLOR_MAP.get)(grid)
    ax.imshow(grid, cmap=None)
    ax.set_xticks([])
    ax.set_yticks([])

    for x in range(grid.shape[1] + 1):
        ax.axvline(x - 0.5, color='black', linewidth=0.5)
    for y in range(grid.shape[0] + 1):
        ax.axhline(y - 0.5, color='black', linewidth=0.5)

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=COLOR_MAP[s["id"]]) for s in SPECIES]
    labels = [s["name"] for s in SPECIES]
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))

    st.pyplot(fig)
    st.success("Simulation complete!")
