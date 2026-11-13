"""Tests for scripts/check_dataset.py on tiny made-up files in a temporary folder."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_dataset.py"


def pair(pair_id, person, a, b, label="contradiction", kind=None, a_date="2025-01-01", b_date="2025-01-01"):
    return {"id": pair_id, "person": person, "a": {"text": a, "written_at": a_date},
            "b": {"text": b, "written_at": b_date}, "label": label, "type": kind, "reason": "Made up."}


TEST = [pair("t-1", "Zed Quorra", "Zed keeps a red kayak at the lake.", "Zed keeps a green kayak at the lake.")]
GOOD = [
    pair("x-1", "Pim Oduya", "Pim bakes rye bread on Sundays.", "Pim has never baked bread."),
    pair("x-2", "Lio Varga", "Lio rents a flat near the harbour.", "Lio moved into a flat near the market.",
         "non_contradiction", "update", b_date="2025-04-01"),
    pair("x-3", "Ude Saro", "Ude's cat might be allergic to fish.", "Ude feeds the cat salmon sometimes.", "unclear"),
]


def write(tmp_path, name, pairs):
    path = tmp_path / name
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    return str(path)


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_good_file_passes_all_walls(tmp_path):
    result = run(write(tmp_path, "train.jsonl", GOOD), "--against", write(tmp_path, "test.jsonl", TEST),
                 "--distinct-from", write(tmp_path, "valid.jsonl", [GOOD[0] | {"id": "v-1", "a": {"text": "Pim swims.", "written_at": "2025-01-01"}}]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unclear: 1" in result.stdout
    assert "PASS" in result.stdout


def test_unclear_with_a_type_fails(tmp_path):
    bad = [GOOD[2] | {"type": "update"}]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "must be null" in result.stderr


def test_test_persons_name_fails(tmp_path):
    bad = [pair("x-9", "Pim Quorra", "Pim walks to work.", "Pim cycles to work.")]
    result = run(write(tmp_path, "train.jsonl", bad), "--against", write(tmp_path, "test.jsonl", TEST))
    assert result.returncode != 0 and "'Quorra' belongs to a person" in result.stderr


def test_near_copy_of_a_test_pair_fails(tmp_path):
    bad = [pair("x-9", "Pim Oduya", "Pim keeps a red kayak at the lake.", "Pim keeps a blue kayak at the lake.")]
    result = run(write(tmp_path, "train.jsonl", bad), "--against", write(tmp_path, "test.jsonl", TEST))
    assert result.returncode != 0 and "near copy" in result.stderr and "t-1" in result.stderr


def test_pair_repeated_in_other_file_fails_even_when_swapped(tmp_path):
    swapped = GOOD[0] | {"id": "v-1", "a": GOOD[0]["b"], "b": GOOD[0]["a"]}
    result = run(write(tmp_path, "train.jsonl", GOOD), "--distinct-from", write(tmp_path, "valid.jsonl", [swapped]))
    assert result.returncode != 0 and "repeats v-1" in result.stderr


def test_banned_name_fails(tmp_path):
    bad = [pair("x-9", "Mara Oduya", "Mara walks to work.", "Mara cycles to work.")]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "banned name" in result.stderr


def test_sentence_template_used_four_times_fails(tmp_path):
    people = ["Ama Kito", "Bel Rusu", "Cai Domo", "Dov Pell"]
    bad = [pair(f"x-{i}", p, f"{p.split()[0]} grew up in Tarno.", f"{p.split()[0]} has a garden.", "non_contradiction",
                "merely_related") for i, p in enumerate(people)]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "sentence pattern used more than 3 times" in result.stderr


def test_unclear_reasons_with_one_start_four_times_fail(tmp_path):
    people = ["Ama Kito", "Bel Rusu", "Cai Domo", "Dov Pell"]
    topics = [("keeps bees on the roof", "is scared of wasps"), ("sings in a choir", "has a sore throat"),
              ("owns a red canoe", "never goes near rivers"), ("repairs old radios", "hates loud noise")]
    bad = [pair(f"x-{i}", p, f"{p.split()[0]} {a}.", f"{p.split()[0]} {b}.", "unclear") | {
           "reason": f"{p.split()[0]}'s case depends on {a.split()[-1]}."} for i, (p, (a, b)) in enumerate(zip(people, topics))]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "4 of 4 unclear reasons in a row start 'NAME case depends'" in result.stderr
    result = run(write(tmp_path, "train.jsonl", bad[:3]))
    assert result.returncode == 0, result.stdout + result.stderr


HARD = [
    GOOD[0] | {"hard": "explain_away"},
    GOOD[1] | {"hard": "indirect_update"},
    pair("x-4", "Oro Banda", "Oro swims on Tuesdays.", "Oro swims on Fridays.", "non_contradiction", "merely_related")
    | {"hard": "rule5_bait"},
]


def test_hard_tags_pass_and_print_counts_per_trap(tmp_path):
    result = run(write(tmp_path, "hard.jsonl", HARD))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "explain_away (contradiction): 1 / 0 / 1" in result.stdout
    assert "indirect_update (update): 1 / 1 / 0" in result.stdout
    assert "rule5_bait (merely_related): 1 / 0 / 1" in result.stdout


def test_one_pair_without_a_hard_tag_fails(tmp_path):
    bad = HARD[:2] + [{k: v for k, v in HARD[2].items() if k != "hard"}]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0 and "line 3: missing field 'hard'" in result.stderr


def test_unknown_hard_tag_fails(tmp_path):
    bad = HARD[:2] + [HARD[2] | {"hard": "sneaky"}]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0 and "hard 'sneaky' is not one of" in result.stderr


def test_hard_tag_that_does_not_fit_its_label_and_type_fails(tmp_path):
    bad = HARD[:2] + [HARD[2] | {"hard": "looks_like_clash"}]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0
    assert "hard 'looks_like_clash' does not fit label 'non_contradiction' with type 'merely_related'" in result.stderr


def test_name_wall_covers_distinct_from_files_for_tagged_files(tmp_path):
    other = [pair("v-1", "Kel Oduya", "Kel paints fences.", "Kel sells paint.", "non_contradiction", "merely_related")]
    result = run(write(tmp_path, "hard.jsonl", HARD), "--distinct-from", write(tmp_path, "valid.jsonl", other))
    assert result.returncode != 0 and "name 'Oduya' belongs to a person in" in result.stderr


def test_files_without_hard_tags_keep_the_old_distinct_from_rule(tmp_path):
    other = [pair("v-1", "Kel Oduya", "Kel paints fences.", "Kel sells paint.", "non_contradiction", "merely_related")]
    result = run(write(tmp_path, "train.jsonl", GOOD), "--distinct-from", write(tmp_path, "valid.jsonl", other))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traps:" not in result.stdout


UNCLEAR_HARD = HARD + [GOOD[2] | {"hard": "true_unclear"}]


def test_true_unclear_tag_passes_on_unclear(tmp_path):
    result = run(write(tmp_path, "hard.jsonl", UNCLEAR_HARD))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "true_unclear (unclear): 1 / 0 / 1" in result.stdout


def test_true_unclear_tag_on_a_contradiction_fails(tmp_path):
    bad = HARD[1:] + [GOOD[0] | {"hard": "true_unclear"}]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0
    assert "hard 'true_unclear' does not fit label 'contradiction' with type None" in result.stderr


PROVENANCE = {"source": "synthetic", "label_by": "claude-code"}


def test_source_and_label_by_on_every_line_pass(tmp_path):
    result = run(write(tmp_path, "hard.jsonl", [p | PROVENANCE for p in UNCLEAR_HARD]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Provenance: every pair has source 'synthetic', label_by 'claude-code'" in result.stdout


def test_files_without_source_or_label_by_print_no_provenance(tmp_path):
    result = run(write(tmp_path, "hard.jsonl", UNCLEAR_HARD))
    assert result.returncode == 0 and "Provenance" not in result.stdout


def test_label_by_missing_on_one_line_fails(tmp_path):
    bad = [p | PROVENANCE for p in HARD[:2]] + [HARD[2] | {"source": "synthetic"}]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0 and "line 3: missing field 'label_by'" in result.stderr


def test_source_alone_on_one_line_makes_every_line_need_both(tmp_path):
    bad = [HARD[0] | {"source": "synthetic"}] + HARD[1:]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0 and "line 1: missing field 'label_by'" in result.stderr


def test_unknown_source_or_label_by_value_fails(tmp_path):
    bad = [p | PROVENANCE for p in HARD[:2]] + [HARD[2] | PROVENANCE | {"label_by": "claude"}]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0 and "label_by 'claude' is not one of ('claude-code',)" in result.stderr
    bad = [HARD[0] | PROVENANCE | {"source": "verdict"}] + [p | PROVENANCE for p in HARD[1:]]
    result = run(write(tmp_path, "hard.jsonl", bad))
    assert result.returncode != 0 and "source 'verdict' is not one of ('synthetic',)" in result.stderr


def test_against_takes_several_test_sets_and_walls_off_each(tmp_path):
    first = write(tmp_path, "test.jsonl", TEST)
    second = write(tmp_path, "test-hard.jsonl", [pair("h-1", "Nim Oduya", "Nim grows chillies.", "Nim hates chillies.")])
    result = run(write(tmp_path, "train.jsonl", GOOD), "--against", first, "--against", second)
    assert result.returncode != 0 and "name 'Oduya' belongs to a person in" in result.stderr and "test-hard.jsonl" in result.stderr
    near = [pair("x-9", "Pim Lund", "Pim grows chillies on the balcony.", "Pim hates chillies on the balcony.")]
    second = write(tmp_path, "test-hard.jsonl", [pair("h-1", "Nim Rask", "Nim grows chillies on the balcony.", "Nim loves chillies on the balcony.")])
    result = run(write(tmp_path, "train.jsonl", near), "--against", first, "--against", second)
    assert result.returncode != 0 and "near copy" in result.stderr and "h-1" in result.stderr
    result = run(write(tmp_path, "train.jsonl", GOOD[1:]), "--against", first, "--against",
                 write(tmp_path, "test-hard.jsonl", [pair("h-1", "Nim Rask", "Nim grows chillies.", "Nim hates chillies.")]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Wall: no names of the 2 name words in") == 2


# Cross-pair check (M3, lesson C1): repeated ideas across all files given.
# The same idea in other words, as in M2's "vegetarian eats meat" pairs; under the near-copy limit (46% shared words).
VEG_A = pair("c-1", "Oro Tamsi", "Oro is a strict vegetarian.", "Oro grills lamb chops for his supper.")
VEG_B = pair("c-2", "Ivo Penner", "Ivo is a strict vegetarian.", "Ivo eats pork sausages for supper.")
OTHERS = [
    pair("c-3", "Ula Brekk", "Ula keeps her canoe in the boathouse.", "Ula paddles across the bay at dawn."),
    pair("c-4", "Eno Valt", "Eno repairs old radios in his shed.", "Eno sells the radios at the flea market."),
    pair("c-5", "Asa Mirro", "Asa learned to weave baskets from reeds.", "Asa's baskets hang above the kitchen window."),
]


def load_checker():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_dataset_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cross_pair_passes_when_ideas_differ(tmp_path):
    result = run(write(tmp_path, "new.jsonl", [VEG_A] + OTHERS), "--cross-pair")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cross-pair check: no two of the 4 pairs in all files given score above 0.44" in result.stdout


def test_cross_pair_fails_on_the_same_idea_in_other_words_in_one_file(tmp_path):
    result = run(write(tmp_path, "new.jsonl", [VEG_A, VEG_B] + OTHERS), "--cross-pair")
    assert result.returncode != 0
    assert "TOO ALIKE" in result.stderr and "(c-1) and c-2 of this file" in result.stderr
    assert "1 pairs of pairs score above the cross-pair limit 0.44" in result.stderr


def test_without_the_option_the_same_file_still_passes(tmp_path):
    result = run(write(tmp_path, "new.jsonl", [VEG_A, VEG_B] + OTHERS))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cross-pair" not in result.stdout


def test_cross_pair_covers_distinct_from_files_and_shows_their_text(tmp_path):
    other = write(tmp_path, "old-train.jsonl", [VEG_B])
    result = run(write(tmp_path, "new.jsonl", [VEG_A] + OTHERS), "--cross-pair", "--distinct-from", other)
    assert result.returncode != 0
    assert f"(c-1) and c-2 of {other}" in result.stderr
    assert "Oro is a strict vegetarian." in result.stderr


def test_cross_pair_names_a_frozen_test_pair_by_id_only(tmp_path):
    frozen = write(tmp_path, "test.jsonl", [VEG_B])
    result = run(write(tmp_path, "new.jsonl", [VEG_A] + OTHERS), "--cross-pair", "--against", frozen)
    assert result.returncode != 0
    assert f"(c-1) and c-2 of {frozen}" in result.stderr
    assert "pork" not in result.stderr and "Ivo" not in result.stderr and "pork" not in result.stdout


def test_cross_pair_does_not_compare_two_other_files_with_each_other(tmp_path):
    first = write(tmp_path, "old-1.jsonl", [VEG_A])
    second = write(tmp_path, "old-2.jsonl", [VEG_B])
    result = run(write(tmp_path, "new.jsonl", OTHERS), "--cross-pair", "--distinct-from", first, "--distinct-from", second)
    assert result.returncode == 0, result.stdout + result.stderr


def test_idea_words_drop_names_and_little_words_cut_endings_and_add_group_words():
    checker = load_checker()
    words = checker.idea_words(VEG_A)
    assert "Oro" not in words and "oro" not in words and "is" not in words and "for" not in words
    assert "MEAT" in words and "NIGHT" not in words and "WEEKDAY" not in words
    assert "grill" in words and "chop" in words and "supper" in words
    assert "MEAT" in checker.idea_words(VEG_B) and "MEAT" not in checker.idea_words(OTHERS[0])


def test_group_words_raise_the_score_of_the_same_idea_in_other_words():
    checker = load_checker()
    rows = [(1, VEG_A), (2, VEG_B)] + [(n, p) for n, p in enumerate(OTHERS, start=3)]
    with_groups = checker.cross_pair_scores(rows, [], limit=0.0)
    checker.IDEA_GROUP_OF.clear()
    without_groups = checker.cross_pair_scores(rows, [], limit=0.0)
    score = lambda found: next(s for s, _, a, _, b, _ in found if {a, b} == {"c-1", "c-2"})
    assert score(with_groups) > checker.CROSS_PAIR_LIMIT > score(without_groups)
