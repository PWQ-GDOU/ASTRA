import unittest

from scripts.clean_benchmark import create_parser


class CleanBenchmarkCliTests(unittest.TestCase):
    def test_validation_ratio_is_configurable(self):
        args = create_parser().parse_args(["--fd", "FD003", "--val-ratio", "0.35"])
        self.assertEqual(args.fd, "FD003")
        self.assertEqual(args.val_ratio, 0.35)

    def test_explicit_dataset_files_are_configurable(self):
        args = create_parser().parse_args(["--dataset-id", "custom-1", "--train-file", "a.txt", "--test-file", "b.txt", "--rul-file", "c.txt"])
        self.assertEqual((args.dataset_id, args.train_file, args.test_file, args.rul_file), ("custom-1", "a.txt", "b.txt", "c.txt"))


if __name__ == "__main__":
    unittest.main()
