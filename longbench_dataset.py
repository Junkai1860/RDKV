import json
import os

import datasets


_DESCRIPTION = """LongBench benchmark dataset."""
_HOMEPAGE = "https://github.com/THUDM/LongBench"
_DEFAULT_ARCHIVE = "<SCRATCH>/hf_datasets/LongBench/data.zip"

task_list = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "multi_news_e",
    "trec_e",
    "triviaqa_e",
    "samsum_e",
    "passage_count_e",
    "passage_retrieval_en_e",
    "lcc_e",
    "repobench-p_e",
]


class LongBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class LongBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [LongBenchConfig(name=task_name) for task_name in task_list]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            features=datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "answers": [datasets.Value("string")],
                    "length": datasets.Value("int32"),
                    "dataset": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "all_classes": [datasets.Value("string")],
                    "_id": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        archive_path = os.environ.get("LONGBENCH_DATA_ARCHIVE", _DEFAULT_ARCHIVE)
        if not os.path.exists(archive_path):
            raise FileNotFoundError(
                f"LongBench archive not found at {archive_path}. "
                "Set LONGBENCH_DATA_ARCHIVE to a local data.zip."
            )
        data_dir = dl_manager.extract(archive_path)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "data", f"{self.config.name}.jsonl")},
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as file:
            for idx, line in enumerate(file):
                item = json.loads(line)
                yield f"{self.config.name}-{idx}", {
                    "input": item["input"],
                    "context": item["context"],
                    "answers": item["answers"],
                    "length": item["length"],
                    "dataset": item["dataset"],
                    "language": item["language"],
                    "_id": item["_id"],
                    "all_classes": item["all_classes"],
                }
