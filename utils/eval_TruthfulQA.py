from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.models.base_model import DeepEvalBaseLLM
import concurrent.futures

from common_utils import get_response, model
from scratchpad import get_scratchpad_response


class CustomModel(DeepEvalBaseLLM):
    def __init__(self, ignore_cache=False, scratchpad=False):
        self.ignore_cache = ignore_cache
        self.scratchpad = scratchpad
        self.counter = 0

    def load_model(self):  # type: ignore
        return True

    def generate(self, prompt: str) -> str:
        self.counter += 1
        print(f"Generating response {self.counter}...")
        if self.scratchpad:
            return get_scratchpad_response(prompt, ignore_cache=self.ignore_cache)  # type: ignore
        return get_response(prompt, ignore_cache=self.ignore_cache)  # type: ignore

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            responses = list(executor.map(self.generate, prompts))
        return responses

    def get_model_name(self):  # type: ignore
        return model


# Define benchmark with specific tasks and shots
tasks=[TruthfulQATask.LOGICAL_FALSEHOOD, TruthfulQATask.MISCONCEPTIONS,
        TruthfulQATask.MISINFORMATION]
benchmark = TruthfulQA(tasks=tasks, mode=TruthfulQAMode.MC2)
scratchpad_benchmark = TruthfulQA(tasks=tasks, mode=TruthfulQAMode.MC2)

if __name__ == "__main__":
    print("\n ----- Standard Evaluation ----- \n")
    benchmark.evaluate(model=CustomModel())
    print("\n ----- Scratchpad Evaluation ----- \n")
    scratchpad_benchmark.evaluate(model=CustomModel(scratchpad=True))

    if not benchmark.overall_score:
        benchmark.overall_score = 0.0
    if not scratchpad_benchmark.overall_score:
        scratchpad_benchmark.overall_score = 0.0

    print("-------- Overall Scores: --------")
    print("Standard Score:", benchmark.overall_score)
    print("Scratchpad Score:", scratchpad_benchmark.overall_score)
    change = scratchpad_benchmark.overall_score - benchmark.overall_score
    print(f"Improvement: +{change:.2f}%")

    print("\n-------- Task-wise Scores: --------")
    print("Standard Evaluation:")
    print(benchmark.task_scores.to_string())
    print("\nScratchpad Evaluation:")
    print(scratchpad_benchmark.task_scores.to_string())
