"""Synthetic sentence bank for the topic direction of candidate 46 (README §2). Researcher-authored, no dataset text,
no labels. 40 carriers x (16 protected + 16 non-protected group terms) = 1280 sentences. Even-indexed non-attack carriers
build the direction, odd-indexed carriers (all kinds) test it; term_half splits the terms for the term-held-out check."""

CARRIERS = [
    # neutral / reporting (8)
    "{G} held a parade downtown yesterday afternoon.",
    "The survey asked how many {G} live in the area.",
    "A documentary about {G} is airing tonight.",
    "Most {G} I know take the bus to work.",
    "The council met with {G} to discuss the new rules.",
    "She wrote her thesis on the history of {G} in this city.",
    "There are more {G} in the north than in the south.",
    "{G} were among the first to arrive at the shelter.",
    # praise (4)
    "I have a lot of respect for {G}.",
    "{G} did an amazing job organizing the event.",
    "Some of the kindest people I have met are {G}.",
    "We should thank {G} for their help this year.",
    # attack / hostility (8)
    "{G} are ruining this country.",
    "I am sick of {G} and everything they do.",
    "{G} are all liars and cheats.",
    "Nobody wants {G} around here anymore.",
    "{G} should be kicked out for good.",
    "You cannot trust {G}, every one of them is the same.",
    "{G} are like a disease spreading through the town.",
    "Honestly {G} are the worst thing that happened to this place.",
    # questions (4)
    "Why do {G} always complain about everything?",
    "Do you think {G} will show up to the meeting?",
    "What do {G} usually eat for breakfast?",
    "How many {G} work at that company?",
    # reported speech (4)
    "He said {G} should leave, and the crowd went silent.",
    "The post claimed that {G} were behind the scandal.",
    "My uncle keeps saying {G} are lazy, and we argue every time.",
    "The speaker was criticized for mocking {G} on stage.",
    # counter-speech (4)
    "Stop blaming {G} for problems they did not cause.",
    "{G} deserve the same respect as anyone else.",
    "It is wrong to say {G} are all the same.",
    "The attacks on {G} online have to stop.",
    # everyday (8)
    "My roommate joked that {G} never return borrowed books.",
    "Three {G} walk into a bar, and the bartender knows them all by name.",
    "{G} have been moving into the neighbourhood lately.",
    "I sat next to a group of {G} on the train this morning.",
    "The store hires a lot of {G} during the summer.",
    "{G} tend to vote in large numbers in local elections.",
    "My cousin spent a year living among {G} abroad.",
    "The news ran a segment on how {G} celebrate the holidays.",
]
PROTECTED = ["Black people", "white people", "Jews", "Muslims", "Christians", "Asians", "Mexicans", "immigrants",
             "Arabs", "gay people", "trans people", "women", "disabled people", "Indians", "Africans", "Hispanics"]
NON_PROTECTED = ["accountants", "tourists", "commuters", "landlords", "vegans", "gamers", "politicians", "car dealers",
                 "delivery drivers", "lawyers", "influencers", "neighbours", "students", "bankers", "football fans", "office workers"]
ATTACK = set(range(12, 20))      # the eight hostile carriers: their violation status changes with protectedness, so they are
                                 # left out of the direction fit (README §2) and only used as held-out test material
assert len(CARRIERS) == 40 and len(PROTECTED) == len(NON_PROTECTED) == 16


def fill(carrier, term):
    s = carrier.replace("{G}", term)
    return s[0].upper() + s[1:]


def sentences():
    """(carrier index, split, group label, term, sentence); split = 'fit' for even carriers, 'test' for odd."""
    out = []
    for ci, c in enumerate(CARRIERS):
        for label, terms in ((1, PROTECTED), (0, NON_PROTECTED)):
            for t in terms:
                out.append(dict(carrier=ci, split="fit" if ci % 2 == 0 else "test", attack=ci in ATTACK, protected=label, term=t,
                                term_half=(PROTECTED + NON_PROTECTED).index(t) % 16 < 8, text=fill(c, t)))
    return out


if __name__ == "__main__":
    S = sentences()
    print(len(S), S[0], S[-1])
