import dataclasses


@dataclasses.dataclass
class DatasetArgs:
    dataset_name: str = dataclasses.field(
        default="",
        metadata={"help": "The name of the dataset."}
    )

    generate_caption: bool = dataclasses.field(
        default=False,
        metadata={"help": "Whether to generate caption."}
    )

    generate_solution: bool = dataclasses.field(
        default=False,
        metadata={"help": "Whether to generate solution."}
    )

    prompt_provide_choices: bool = dataclasses.field(
        default=False,
        metadata={"help": "Whether to provide choices in prompt."}
    )

    def check_args_validity(self):
        assert not (self.generate_caption and self.generate_solution), "Cannot generate both caption and solution."
        assert not (self.generate_caption and self.prompt_provide_choices), "Cannot generate caption and provide choices in prompt."
        assert self.generate_caption or self.generate_solution, "Must generate caption or solution."
