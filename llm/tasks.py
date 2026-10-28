"""50 auto-graded tasks: 20 coding (hidden unit tests), 15 analytical (exact number), 15 research
(invented facts, keyword retrieval). Every coding task ships a reference solution so graders can be tested."""
import math
import os
import re
import subprocess
import sys
import tempfile
from functools import reduce

# ---------------------------------------------------------------- coding
# (fn, spec, reference, visible asserts, hidden asserts)
_C = []


def _code(fn, spec, ref, visible, hidden):
    _C.append(dict(kind="coding", fn=fn, spec=spec, ref=ref, visible=visible, hidden=hidden))


_code("second_largest", "Return the second largest DISTINCT value in the list nums, or None if there is none.",
      "def second_largest(nums):\n    s = sorted(set(nums), reverse=True)\n    return s[1] if len(s) > 1 else None",
      ["second_largest([3,1,2]) == 2", "second_largest([5,5]) is None"],
      ["second_largest([1,2]) == 1", "second_largest([10,9,9,8]) == 9", "second_largest([]) is None",
       "second_largest([-1,-5,-3]) == -3", "second_largest([7]) is None", "second_largest([2,2,3,3]) == 2"])
_code("is_palindrome", "Return True if string s reads the same forwards and backwards, ignoring case and every non-alphanumeric character.",
      "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
      ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('hello') is False"],
      ["is_palindrome('') is True", "is_palindrome('No lemon, no melon') is True", "is_palindrome('ab') is False",
       "is_palindrome('12321') is True", "is_palindrome('0P') is False", "is_palindrome('Was it a car or a cat I saw?') is True"])
_code("fizzbuzz", "Return a list of strings for the numbers 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number as a string.",
      "def fizzbuzz(n):\n    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]",
      ["fizzbuzz(3) == ['1','2','Fizz']", "fizzbuzz(5) == ['1','2','Fizz','4','Buzz']"],
      ["fizzbuzz(15)[-1] == 'FizzBuzz'", "fizzbuzz(0) == []", "fizzbuzz(1) == ['1']", "fizzbuzz(10)[8] == 'Fizz'",
       "len(fizzbuzz(30)) == 30", "fizzbuzz(6)[5] == 'Fizz'"])
_code("count_vowels", "Return how many vowels (a, e, i, o, u, case-insensitive) the string s contains. 'y' is not a vowel.",
      "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())",
      ["count_vowels('Hello') == 2", "count_vowels('xyz') == 0"],
      ["count_vowels('AEIOU') == 5", "count_vowels('') == 0", "count_vowels('rhythm') == 0",
       "count_vowels('Programming Language') == 7", "count_vowels('y') == 0"])
_code("flatten", "Flatten an arbitrarily nested list of lists into one flat list, keeping left-to-right order.",
      "def flatten(lst):\n    out = []\n    for x in lst:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out",
      ["flatten([1,[2,3]]) == [1,2,3]", "flatten([]) == []"],
      ["flatten([[1,[2]],[[3]],4]) == [1,2,3,4]", "flatten([[],[[]]]) == []", "flatten([1,[2,[3,[4,[5]]]]]) == [1,2,3,4,5]",
       "flatten(['a',['b']]) == ['a','b']", "flatten([[1],[2],[3]]) == [1,2,3]"])
_code("rle", "Run-length encode string s: each run of the same character becomes the character followed by the run length, e.g. 'aaabcc' -> 'a3b1c2'. Empty string gives ''.",
      "def rle(s):\n    out, i = '', 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out += s[i] + str(j - i)\n        i = j\n    return out",
      ["rle('aaabcc') == 'a3b1c2'", "rle('') == ''"],
      ["rle('a') == 'a1'", "rle('aabb') == 'a2b2'", "rle('abc') == 'a1b1c1'", "rle('zzzzzzzzzzzz') == 'z12'", "rle('aaAA') == 'a2A2'"])
_code("roman_to_int", "Convert a Roman numeral string (I, V, X, L, C, D, M with subtractive pairs like IV, IX, XL, XC, CD, CM) to an integer.",
      "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t = 0\n    for i, c in enumerate(s):\n        t += -v[c] if i + 1 < len(s) and v[c] < v[s[i+1]] else v[c]\n    return t",
      ["roman_to_int('III') == 3", "roman_to_int('IV') == 4"],
      ["roman_to_int('IX') == 9", "roman_to_int('LVIII') == 58", "roman_to_int('MCMXCIV') == 1994",
       "roman_to_int('XL') == 40", "roman_to_int('MMXXIV') == 2024", "roman_to_int('CDXLIV') == 444"])
_code("valid_brackets", "Return True if string s contains only correctly nested and closed brackets of the kinds (), [] and {}. The empty string is valid.",
      "def valid_brackets(s):\n    pairs = {')':'(', ']':'[', '}':'{'}\n    st = []\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif not st or st.pop() != pairs[c]:\n            return False\n    return not st",
      ["valid_brackets('()[]{}') is True", "valid_brackets('(]') is False"],
      ["valid_brackets('') is True", "valid_brackets('([{}])') is True", "valid_brackets('([)]') is False",
       "valid_brackets('((') is False", "valid_brackets('}') is False", "valid_brackets('{[]}') is True"])
_code("two_sum", "Return the tuple (i, j) with i < j and nums[i] + nums[j] == target. If several pairs work, return the one with the smallest i, then the smallest j. Return None if no pair exists.",
      "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return (i, j)\n    return None",
      ["two_sum([2,7,11,15], 9) == (0,1)", "two_sum([1,2], 10) is None"],
      ["two_sum([3,2,4], 6) == (1,2)", "two_sum([3,3], 6) == (0,1)", "two_sum([1,5,5,1], 6) == (0,1)",
       "two_sum([0,4,3,0], 0) == (0,3)", "two_sum([], 1) is None", "two_sum([-1,-2,-3,-4,-5], -8) == (2,4)"])
_code("merge_intervals", "Merge overlapping intervals (intervals that touch, like [1,4] and [4,5], also merge). Input is a list of [start, end] lists in any order; return a list of [start, end] lists sorted by start.",
      "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
      ["merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]", "merge_intervals([]) == []"],
      ["merge_intervals([[1,4],[4,5]]) == [[1,5]]", "merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]",
       "merge_intervals([[1,10],[2,3]]) == [[1,10]]", "merge_intervals([[1,2]]) == [[1,2]]",
       "merge_intervals([[2,3],[4,5],[6,7],[8,9],[1,10]]) == [[1,10]]"])
_code("group_anagrams", "Group words that are anagrams of each other. Return a list of groups; sort the words inside each group alphabetically and sort the groups by their first word.",
      "def group_anagrams(words):\n    d = {}\n    for w in words:\n        d.setdefault(''.join(sorted(w)), []).append(w)\n    return sorted((sorted(g) for g in d.values()), key=lambda g: g[0])",
      ["group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]", "group_anagrams([]) == []"],
      ["group_anagrams(['a']) == [['a']]", "group_anagrams(['ab','ba','abc']) == [['ab','ba'],['abc']]",
       "group_anagrams(['x','x']) == [['x','x']]", "group_anagrams(['listen','silent','enlist','google']) == [['enlist','listen','silent'],['google']]"])
_code("longest_common_prefix", "Return the longest common prefix of a list of strings, or '' if there is none or the list is empty.",
      "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(p):\n            p = p[:-1]\n    return p",
      ["longest_common_prefix(['flower','flow','flight']) == 'fl'", "longest_common_prefix(['dog','car']) == ''"],
      ["longest_common_prefix([]) == ''", "longest_common_prefix(['a']) == 'a'", "longest_common_prefix(['abc','abc']) == 'abc'",
       "longest_common_prefix(['ab','a']) == 'a'", "longest_common_prefix(['interview','internet','interval']) == 'inter'"])
_code("rotate_right", "Return a new list: lst rotated to the right by k places. k may be 0, larger than len(lst), or negative (negative means rotate left). An empty list stays empty.",
      "def rotate_right(lst, k):\n    if not lst:\n        return []\n    k %= len(lst)\n    return lst[-k:] + lst[:-k] if k else list(lst)",
      ["rotate_right([1,2,3,4,5], 2) == [4,5,1,2,3]", "rotate_right([1,2], 0) == [1,2]"],
      ["rotate_right([1,2,3], 3) == [1,2,3]", "rotate_right([1,2,3], 4) == [3,1,2]", "rotate_right([], 5) == []",
       "rotate_right([1], 100) == [1]", "rotate_right([1,2,3,4], -1) == [2,3,4,1]"])
_code("lcm_list", "Return the least common multiple of a list of positive integers. For an empty list return 1.",
      "from math import gcd\ndef lcm_list(nums):\n    r = 1\n    for n in nums:\n        r = r * n // gcd(r, n)\n    return r",
      ["lcm_list([4,6]) == 12", "lcm_list([3]) == 3"],
      ["lcm_list([]) == 1", "lcm_list([2,3,5]) == 30", "lcm_list([12,18]) == 36", "lcm_list([7,7]) == 7",
       "lcm_list([1,2,3,4,5,6]) == 60", "lcm_list([10,15,25]) == 150"])
_code("binary_search", "Return the index of x in the sorted list arr, or -1 if it is absent. If x appears several times, return the LEFTMOST index.",
      "def binary_search(arr, x):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        m = (lo + hi) // 2\n        if arr[m] < x:\n            lo = m + 1\n        else:\n            hi = m\n    return lo if lo < len(arr) and arr[lo] == x else -1",
      ["binary_search([1,3,5,7], 5) == 2", "binary_search([1,3], 2) == -1"],
      ["binary_search([], 1) == -1", "binary_search([2,2,2,2], 2) == 0", "binary_search([1,2,2,3], 2) == 1",
       "binary_search([1,2,3], 3) == 2", "binary_search([1,2,3], 0) == -1", "binary_search([5], 5) == 0"])
_code("max_subarray", "Return the largest sum of any non-empty contiguous sub-list of nums.",
      "def max_subarray(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
      ["max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", "max_subarray([1]) == 1"],
      ["max_subarray([-3,-1,-2]) == -1", "max_subarray([5,4,-1,7,8]) == 23", "max_subarray([0,0]) == 0",
       "max_subarray([-1]) == -1", "max_subarray([2,-1,2]) == 3"])
_code("top_words", "Lowercase text and split it into words made only of the letters a-z (every other character is a separator). Return the k most frequent words as a list, most frequent first; break ties alphabetically.",
      "import re\nfrom collections import Counter\ndef top_words(text, k):\n    c = Counter(re.findall('[a-z]+', text.lower()))\n    return [w for w, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:k]]",
      ["top_words('a b a c b a', 2) == ['a','b']", "top_words('', 3) == []"],
      ["top_words('The cat. The dog! the end', 1) == ['the']", "top_words('b a', 2) == ['a','b']",
       "top_words('x y z', 5) == ['x','y','z']", "top_words('Hi hi HI there', 2) == ['hi','there']",
       "top_words(\"don't stop\", 3) == ['don','stop','t']"])
_code("transpose", "Return the transpose of a matrix given as a list of rows. The empty matrix [] transposes to [].",
      "def transpose(m):\n    return [list(r) for r in zip(*m)]",
      ["transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]", "transpose([]) == []"],
      ["transpose([[1]]) == [[1]]", "transpose([[1,2]]) == [[1],[2]]", "transpose([[1],[2],[3]]) == [[1,2,3]]",
       "transpose([[1,2],[3,4]]) == [[1,3],[2,4]]"])
_code("dedupe", "Return a new list with duplicates removed, keeping the first occurrence of each item and the original order. Items may be of mixed types (1 and '1' are different).",
      "def dedupe(lst):\n    out = []\n    for x in lst:\n        if not any(type(x) == type(y) and x == y for y in out):\n            out.append(x)\n    return out",
      ["dedupe([1,2,1,3]) == [1,2,3]", "dedupe([]) == []"],
      ["dedupe(['a','b','a']) == ['a','b']", "dedupe([1,1,1]) == [1]", "dedupe([3,2,1]) == [3,2,1]",
       "dedupe([1,'1',1]) == [1,'1']", "dedupe([None,None]) == [None]"])
_code("valid_ipv4", "Return True if string s is a valid IPv4 address: exactly four dot-separated parts, each made only of digits, each between 0 and 255, and no leading zeros (the part '0' itself is fine).",
      "def valid_ipv4(s):\n    p = s.split('.')\n    return len(p) == 4 and all(x.isdigit() and x.isascii() and int(x) <= 255 and (x == '0' or not x.startswith('0')) for x in p)",
      ["valid_ipv4('192.168.1.1') is True", "valid_ipv4('256.1.1.1') is False"],
      ["valid_ipv4('0.0.0.0') is True", "valid_ipv4('01.1.1.1') is False", "valid_ipv4('1.1.1') is False",
       "valid_ipv4('1.1.1.1.1') is False", "valid_ipv4('a.b.c.d') is False", "valid_ipv4('255.255.255.255') is True",
       "valid_ipv4('1.2.3.-4') is False", "valid_ipv4('') is False"])


def run_asserts(code, asserts, timeout=10):
    """Run model-written code + asserts in a separate Python process. Returns (n_passed, n_total)."""
    body = code + "\n\n_ok = 0\n" + "".join(
        f"try:\n    assert {a}\n    _ok += 1\nexcept BaseException:\n    pass\n" for a in asserts) + "print('PASSED', _ok)\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        open(path, "w").write(body)
        try:
            out = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                                 timeout=timeout, cwd=d).stdout
            m = re.search(r"PASSED (\d+)", out)
            return (int(m.group(1)) if m else 0), len(asserts)
        except subprocess.TimeoutExpired:
            return 0, len(asserts)


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


# ---------------------------------------------------------------- analytical
_A = [
    ("A shop sells pens at 12 rupees each. Ravi buys 7 pens and pays with a 100 rupee note. How much change does he get?", 100 - 7 * 12),
    ("A jacket costs 2400 rupees. It is sold at 25% discount, then a 5% tax is added on the discounted price. What is the final price?", 2400 * 0.75 * 1.05),
    ("A car travels 150 km at 50 km/h, then 120 km at 80 km/h. What is its average speed for the whole trip in km/h?", (150 + 120) / (150 / 50 + 120 / 80)),
    ("Simple interest on 8000 rupees at 7.5% per year for 4 years is how many rupees?", 8000 * 0.075 * 4),
    ("Two friends share 2600 rupees in the ratio 5:8. How many rupees does the one with the larger share get?", 2600 * 8 / 13),
    ("The average of five numbers is 18. Four of the numbers are 12, 20, 25 and 15. What is the fifth number?", 18 * 5 - (12 + 20 + 25 + 15)),
    ("Pipe A fills a tank in 12 hours and pipe B fills it in 4 hours. Working together, how many hours do they need to fill the empty tank?", 12 * 4 / 16),
    ("A father is 4 times as old as his son. The sum of their ages is 55. How old is the son?", 55 / 5),
    ("A rectangle has perimeter 54 cm and length 17 cm. What is its area in square cm?", 17 * (27 - 17)),
    ("Anu buys 6 notebooks at 45 rupees and 4 pens at 18 rupees each. She gets 10% off the total. How many rupees does she pay?", (6 * 45 + 4 * 18) * 0.9),
    ("A bacteria colony doubles every 3 hours. It starts with 50 bacteria. How many are there after 15 hours?", 50 * 2 ** 5),
    ("A train moves at 72 km/h. How many metres does it travel in 25 seconds?", 72 / 3.6 * 25),
    ("The sum of three consecutive integers is 132. What is the largest of them?", 132 / 3 + 1),
    ("A price rises by 20% and then falls by 20%. If the original price was 500 rupees, what is the final price?", 500 * 1.2 * 0.8),
    ("Two trains start 330 km apart and move towards each other at 45 km/h and 65 km/h. After how many hours do they meet?", 330 / 110),
]


def grade_number(text, answer, tol=0.011):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return bool(nums) and any(abs(float(n) - answer) <= tol for n in nums[-2:])


# ---------------------------------------------------------------- research (invented facts)
DOCS = [
    "Veltrix Corp was founded in 1987 by Marta Oyelaran in the city of Quillhaven. Its flagship product is the Brannock drill.",
    "Marta Oyelaran studied metallurgy at Dunmore Institute before founding her company. She retired in 2009 and moved to Port Selby.",
    "The Brannock drill weighs 4.2 kilograms and runs on a 36 volt battery. It was first released in 1991.",
    "Quillhaven is a harbor city with a population of 212,000. Its mayor since 2018 is Tobias Wrenfield.",
    "Tobias Wrenfield previously worked as a ferry pilot. He was born in the village of Aldermoor.",
    "Aldermoor is known for its annual lantern festival, held every October.",
    "Norvane Labs was founded in 2003 by Ilse Karimov in the city of Dunmore. Its main product is the Sable microscope.",
    "The Sable microscope magnifies up to 900 times and costs 1,450 euros. It was released in 2006.",
    "Dunmore Institute is a technical university in Dunmore, established in 1952. Its current rector is Priya Halvorsen.",
    "Ilse Karimov won the Harlow Prize in 2014 for her work on lens coatings.",
    "The Harlow Prize is awarded each year in Copenhagen and comes with a grant of 80,000 euros.",
    "Port Selby is a fishing town on the Marrow coast. The town's lighthouse was built in 1874.",
    "Veltrix Corp opened a second factory in 1999 in the town of Fenwick, employing 340 people.",
    "Fenwick sits on the River Orrel and is 60 kilometers from Quillhaven.",
    "Kestrel Dynamics, founded in 1995 by Omar Teague in Fenwick, makes the Ardent drill, which weighs 5.1 kilograms.",
    "Omar Teague left Kestrel Dynamics in 2011 and now teaches at Dunmore Institute.",
]
# (question, accepted answers (normalised substrings), forbidden substrings, gold doc ids)
_R = [
    ("Who founded Veltrix Corp?", ["oyelaran"], [], [0]),
    ("In which year was the Brannock drill first released?", ["1991"], [], [2]),
    ("What is the population of Quillhaven?", ["212000"], [], [3]),
    ("Who is the mayor of the city where Veltrix Corp was founded?", ["wrenfield"], [], [0, 3]),
    ("In which village was the mayor of Quillhaven born?", ["aldermoor"], [], [3, 4]),
    ("When is the lantern festival held in the village where Tobias Wrenfield was born?", ["october"], [], [4, 5]),
    ("At which institute did the founder of Veltrix Corp study?", ["dunmore institute"], [], [0, 1]),
    ("Who is the rector of the institute where Marta Oyelaran studied?", ["halvorsen"], [], [0, 1, 8]),
    ("How many euros is the grant of the prize won by the founder of Norvane Labs?", ["80000"], [], [6, 9, 10]),
    ("In which city is the Harlow Prize awarded?", ["copenhagen"], [], [10]),
    ("How many euros does the Sable microscope cost?", ["1450"], [], [7]),
    ("Which drill is heavier, the Brannock or the Ardent? Answer with one word.", ["ardent"], ["brannock"], [2, 14]),
    ("How many people work at the second factory of Veltrix Corp?", ["340"], [], [12]),
    ("How many kilometers is the town of Veltrix Corp's second factory from Quillhaven?", ["60"], [], [12, 13]),
    ("Where does the founder of Kestrel Dynamics teach now?", ["dunmore institute"], [], [14, 15]),
]


def _toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def retrieve(query, exclude=(), k=2):
    """Keyword overlap with a mild length-free idf. Returns doc ids not in `exclude`."""
    df = {}
    for d in DOCS:
        for t in set(_toks(d)):
            df[t] = df.get(t, 0) + 1
    q = set(_toks(query))
    scored = sorted(((sum(math.log(1 + len(DOCS) / df[t]) for t in q & set(_toks(d))), i)
                     for i, d in enumerate(DOCS) if i not in exclude), reverse=True)
    return [i for s, i in scored[:k] if s > 0]


def grade_text(text, accept, forbid):
    t = text.lower().replace(",", "")
    return all(a in t for a in accept) and not any(f in t for f in forbid)


# ---------------------------------------------------------------- the task list
def all_tasks():
    tasks = []
    for i, c in enumerate(_C):
        tasks.append(dict(id=f"code{i:02d}", type="coding", **{k: c[k] for k in ("fn", "spec", "ref", "visible", "hidden")}))
    for i, (q, a) in enumerate(_A):
        tasks.append(dict(id=f"ana{i:02d}", type="analytical", question=q, answer=float(a)))
    for i, (q, acc, forb, gold) in enumerate(_R):
        tasks.append(dict(id=f"res{i:02d}", type="research", question=q, accept=acc, forbid=forb, gold=gold))
    return tasks


def grade(task, output):
    """True if the final output is correct. Coding uses HIDDEN tests."""
    if task["type"] == "coding":
        ok, n = run_asserts(extract_code(output), task["visible"] + task["hidden"])
        return ok == n
    if task["type"] == "analytical":
        return grade_number(output, task["answer"])
    return grade_text(output, task["accept"], task["forbid"])
