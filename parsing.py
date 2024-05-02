def normalize_ans(a):
    a = a.lower().strip(' <>*().?,\'\"')
    a = a.replace('  ', ' ')
    a = a.replace('  ', ' ')
    
    if a.startswith('a '):
        a = a[2:]
    if a.startswith('an '):
        a = a[3:]
    if a.startswith('the '):
        a = a[4:]

    return a

def normalize_sample(a):
    a = a.lower().strip(' <>*().!?,\'\"')
    if a.startswith("a "):
        a = a[2:]
    if a.startswith("an "):
        a = a[3:]
    if a.startswith("the "):
        a = a[4:]
    if a.endswith(" and"):
        a = a[:-4]
    if a.startswith("yes,") or a.startswith("yes!") or a.startswith("yes "):
        a = a[:3]
    if a.startswith("no,") or a.startswith("no!") or a.startswith("no "):
        a = a[:2]
    
    a = a.replace(",", " ")
    a = a.replace(" the ", " ")
    a = a.replace(" a ", " ")
    a = a.replace(" an ", " ")
    a = a.replace("  ", " ")
    a = a.replace("  ", " ")
    a = a.replace("-", "")
    a = a.replace("'", "")
    a = a.replace("\"", "")
    a = a.replace(".", " ")
    a = a.replace("?", " ")
    a = a.replace("!", " ")
    a = a.replace("\n", " ")
    a = a.replace("  ", " ")
    a = a.replace("  ", " ")
    a = a.strip()

    return a 


def group_ans_simple(guesses):
    # input: 1d list of answers 
    # normalize guesses first
    guesses = [normalize_ans(g) for g in guesses]
    # groups answers by string equality
    guesses_and_probs = sorted(set([(g, guesses.count(g) / len(guesses)) for g in set(guesses)]), key=lambda x: x[1], reverse=True)
    guesses, probs = zip(*guesses_and_probs)
    return list(guesses), list(probs)


def answers_are_equivalent_heuristic(a, b):
    # assume a, b already normalized
    a_words = a.split(" ")
    b_words = b.split(" ")
    # if every word in a is in b or vice versa, return true
    if a == b:
        return True 
    if all([w in b_words for w in a_words]) and all([w in a_words for w in b_words]):
        return True 
    if a.endswith(b) and a[-len(b)-1:-len(b)] != ",":
        return True
    if b.endswith(a) and b[-len(a)-1:-len(a)] != ",": 
        return True
    if a == b + "s":
        return True
    if b == a + "s":
        return True
    return False


def add_to_equiv_class_heuristic(equiv_classes, ans, conf):
    # assume ans already normalized
    for key in equiv_classes.keys():
        if answers_are_equivalent_heuristic(ans, key):
            equiv_classes[key] += conf
            return
    equiv_classes[ans] = conf


def build_equiv_class_heuristics(answers, confidences):
    equiv_classes = {}
    # print("answers, confidences", answers, confidences)
    for ans, conf in zip(answers, confidences):
        # if not unique list, conf is just ans_count 
        # print("e, a, c", equiv_classes, ans, conf)
        add_to_equiv_class_heuristic(equiv_classes, ans, conf)
    # assert that sum of the values in equiv class = 1
    return equiv_classes
