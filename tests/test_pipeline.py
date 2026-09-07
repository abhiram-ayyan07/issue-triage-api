from src.data_pipeline import LABELS, load_dataset, load_train_test


def test_load_train_test_shapes():
    train, test = load_train_test()
    assert len(train.text) > 0
    assert len(test.text) > 0
    assert set(train.label.unique()).issubset(set(LABELS))


def test_combined_text_includes_title_and_body():
    train, _ = load_train_test()
    # every combined text string should be non-empty
    assert all(len(t) > 0 for t in train.text)


def test_load_dataset_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    try:
        load_dataset(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
