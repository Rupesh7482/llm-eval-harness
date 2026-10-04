import json

golden = json.load(open("data/golden.json"))
chunks = json.load(open("data/chunks.json"))
by_id = {item["id"]: item for item in golden}

by_id[1]["question"] = "What is natural acceptance?"

by_id[2]["question"] = "What are the characteristics of natural acceptance?"
by_id[2]["expected_answer"] = (
    "Natural acceptance does not change with time, place or the individual. "
    "It is uncorrupted by likes, dislikes, assumptions or beliefs, it is innate, and it is definite."
)
by_id[2]["context"] = chunks[3][-350:-100] + chunks[4]  # adds the missing first half of the list

by_id[3]["question"] = "What are the guidelines for the content of a course on value education?"
by_id[3]["expected_answer"] = (
    "The content should be universal, rational, natural and verifiable, "
    "all encompassing, and leading to harmony."
)
by_id[3]["context"] = chunks[7] + chunks[8][100:]  # joins the two chunks without the overlap

by_id[4]["question"] = "What are the two parts of the process of self-exploration?"
by_id[4]["expected_answer"] = (
    "The first part is verifying the proposal on the basis of your own natural acceptance. "
    "The second part is experiential validation, which means trying to live according to the proposal."
)

by_id[7]["expected_answer"] = "The levels are individual, family, society and nature - existence."

by_id[9]["expected_answer"] = (
    "Prosperity requires identifying the required quantity of physical facilities and ensuring "
    "more than that is available. The feeling of prosperity can only be assured if there is "
    "a limit to the need for physical facilities."
)

final = [item for item in golden if item["id"] != 8]  # drop Q8, it makes no sense on its own
with open("data/golden.json", "w") as f:
    json.dump(final, f, indent=2)
print(len(final), "items saved")