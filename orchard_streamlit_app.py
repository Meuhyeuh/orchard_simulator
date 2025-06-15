    species_counts = {s["name"]: 0 for s in SPECIES}
    for i in range(num_rows):
        for j in range(num_cols):
            species_name = ID_TO_NAME[grid[i, j]]
            species_counts[species_name] += 1

    total_trees = sum(species_counts.values())
    species_df = pd.DataFrame.from_dict(species_counts, orient='index', columns=['Tree Count'])
    species_df['Percentage'] = (species_df['Tree Count'] / total_trees * 100).round(1).astype(str) + '%'

    st.subheader("Tree Counts by Species")
    st.write(species_df)
