# Create a mapping for more readable criteria names
criteria_mapping = {
    'crit_Adaptability': 'Adaptability',
    'crit_Consciousness': 'Consciousness',
    'crit_Creativity': 'Creativity',
    'crit_Embodiment': 'Embodiment',
    'crit_Env._interaction': 'Environmental\nInteraction',
    'crit_Generalization': 'Generalization',
    'crit_Goal_Ach.': 'Goal\nAchievement',
    'crit_Knowledge_acq.': 'Knowledge\nAcquisition',
    'crit_Perception': 'Perception',
    'crit_Planning': 'Planning',
    'crit_Problem_solving': 'Problem\nSolving',
    'crit_Reasoning': 'Reasoning',
    'crit_Understanding': 'Understanding'
}

# Create a mapping of long names to shorter versions
label_mapping = {
    'Natural Language Processing': 'NLP',
    'Machine Learning': 'ML',
    'Neuroscience': 'Neuroscience',
    'Computational Linguistics': 'Comp. Linguistics',
    'Artificial Intelligence': 'AI',
    'Cognitive Science': 'Cog. Science',
    'Human-Computer Interaction': 'HCI',
    'Psychology (cognitive, developmental?)': 'Psychology',
    'Computational Social Science': 'Comp. Social Sci.',
    'Mathematics': 'Math',
    'Information Science': 'Info. Science',
    'AI Ethics/Governance': 'AI Ethics'
}

# Create a mapping for occupation names
occupation_mapping = {
    'occ_Academia': 'Academia',
    'occ_Industry': 'Industry',
    'occ_Government_non-profit': 'Government/\nNon-profit',
    'occ_Undisclosed': 'Undisclosed'
}

# Create a comprehensive mapping of regions to their approximate centers (latitude, longitude)
region_centers = {
    'Europe: Northern': (62, 15),      # Centered over Scandinavia
    'Europe: Western': (48, 7),        # Centered over Switzerland
    'Europe: Eastern': (52, 20),       # Centered over Poland
    'Europe: Southern': (41, 12),      # Centered over Italy
    'Europe: Central': (50, 14),       # Centered over Czech Republic
    'Asia: Eastern': (35, 115),        # Centered over East Asia
    'Asia: Southern': (20, 77),        # Centered over India
    'Asia: Western': (35, 40),         # Centered over Middle East
    'Asia: Central': (45, 65),         # Centered over Central Asia
    'Asia: South-Eastern': (10, 106),  # Centered over Southeast Asia
    'Africa: Northern': (30, 10),      # Centered over North Africa
    'Africa: Southern': (-25, 25),     # Centered over South Africa
    'Africa: Eastern': (0, 35),        # Centered over East Africa
    'Africa: Western': (10, 0),        # Centered over West Africa
    'Africa: Central': (0, 20),        # Centered over Central Africa
    'America: Northern': (45, -100),  # Centered over North America
    'America: Central': (15, -90),    # Centered over Central America
    'America: South': (-15, -60),  # Centered over South America
    'Oceania': (-25, 135)             # Centered over Australia
}