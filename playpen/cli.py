import argparse
import inspect
import importlib.util as importlib_util
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Callable, List
from datetime import datetime

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry, GameSpec
from playpen import BasePlayPen
from pathlib import Path
from .data_utils import create_conversational_dataset_for


def train(file_path: str, learner: ModelSpec, teacher: ModelSpec, temperature: float, max_tokens: int):
    def is_playpen(obj):
        return inspect.isclass(obj) and issubclass(obj, BasePlayPen) and obj is not BasePlayPen

    try:
        file_name = os.path.splitext(file_path)[0]
        spec = importlib_util.spec_from_file_location(file_name, file_path)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        playpen_subclasses = inspect.getmembers(module, predicate=is_playpen)
        if len(playpen_subclasses) == 0:
            raise ValueError(f"Cannot load playpen trainer, because no BasePlayPen found in {file_path}.\n"
                             f"Make sure that you have implemented a subclass of BasePlayPen and try again.")
        _, playpen_cls = playpen_subclasses[0]
    except Exception as e:
        raise RuntimeError(f"Cannot load playpen trainer, because {e}")

    game_registry = GameRegistry.from_directories_and_cwd_files()
    model_registry = ModelRegistry.from_packaged_and_cwd_files()

    learner_spec = model_registry.get_first_model_spec_that_unify_with(learner)
    print(f"Found registered model spec that unifies with {learner.to_string()} -> {learner_spec}")

    model_specs = [learner_spec]
    if teacher is not None:
        teacher_spec = model_registry.get_first_model_spec_that_unify_with(learner)
        print(f"Found registered model spec that unifies with {teacher.to_string()} -> {teacher_spec}")
        model_specs.append(teacher_spec)

    backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    for model_spec in model_specs:
        backend_selector = model_spec.backend
        if not backend_registry.is_supported(backend_selector):
            raise ValueError(f"Specified model backend '{backend_selector}' not found in backend registry.")
        print(f"Found registry entry for backend {backend_selector} "
              f"-> {backend_registry.get_first_file_matching(backend_selector)}")

    models = []
    for model_spec in model_specs:  # only now since model loading might take long
        print(f"Dynamically import backend {model_spec.backend}")
        backend = backend_registry.get_backend_for(model_spec.backend)
        model = backend.get_model_for(model_spec)
        model.set_gen_args(max_tokens=max_tokens, temperature=temperature)
        print(f"Successfully loaded {model_spec.model_name} model")
        models.append(model)

    learner_model = models[0]
    if len(models) == 1:
        playpen_cls(learner_model).learn(game_registry)
    else:
        teacher_model = models[1]
        playpen_cls(learner_model, teacher_model).learn(game_registry)


def store_eval_score(file_path: Path, name: str, value):
    try:  # first, try to load file to not overwrite already written eval scores
        with open(file_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"Update {file_path}")
    except FileNotFoundError:
        print(f"Create {file_path}")
        scores = {}
    new_scores = {**scores, **{name: value}}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(new_scores, f)
    print(json.dumps(new_scores, indent=2))
    return new_scores


def to_task_selector(dataset) -> Callable[[str, str], List[int]]:
    import collections
    tasks_by_group = collections.defaultdict(list)
    for row in dataset:  # a list of rows with game, experiment, task_id columns
        key = (row['game'], row['experiment'])
        tasks_by_group[key].append(int(row['task_id']))
    return lambda game, experiment: tasks_by_group[(game, experiment)]


def get_default_results_dir():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_dir = Path("playpen-eval") / timestamp
    return results_dir


def evaluate_suite(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
                   dataset_name: str):
    suite_results_dir = str(results_dir / suite)
    if dataset_name is not None:
        from datasets import load_dataset
        dataset = load_dataset("colab-potsdam/playpen-data", dataset_name, split="validation")
        task_selector = to_task_selector(dataset)
        clem.run(game_selector, [model_spec],
                 gen_args=gen_args, results_dir=suite_results_dir, task_selector=task_selector)
    clem.score(game_selector, suite_results_dir)
    clem.transcripts(game_selector, suite_results_dir)
    df = clem.clemeval.perform_evaluation(suite_results_dir, return_dataframe=True)
    clem_score = df["-, clemscore"][0]
    return clem_score


def evaluate(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
             skip_gameplay: bool):
    overall_results_file = results_dir / f"{model_spec.model_name}.val.json"
    if suite in ["all", "clem"]:
        dataset_name = None if skip_gameplay else "instances"
        game_selector = GameSpec.from_dict({"benchmark": ["2.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        clem_score = evaluate_suite("clem", model_spec, gen_args, results_dir, game_selector, dataset_name)
        store_eval_score(overall_results_file, "clemscore", clem_score)
    if suite in ["all", "static"]:
        dataset_name = None if skip_gameplay else "instances-static"
        game_selector = GameSpec.from_dict({"benchmark": ["static_1.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        stat_score = evaluate_suite("static", model_spec, gen_args, results_dir, game_selector, dataset_name)
        store_eval_score(overall_results_file, "statscore", stat_score)

def collect_failures(results_dir, failures_dir):
    for f in os.listdir(results_dir):
        file_path = os.path.join(results_dir, f)
        
        for game in os.listdir(file_path):
            game_path = os.path.join(file_path, game)
            if not os.path.isdir(game_path):
                continue
            for exp_dir in Path(game_path).iterdir():
                if not os.path.isdir(exp_dir):
                    continue
                for episode_dir in Path(exp_dir).iterdir():
                    if not os.path.isdir(episode_dir):
                        continue
                    scores_path = episode_dir / "scores.json"
                    with open(scores_path, "r") as f:
                        try:
                            data = json.load(f)
                            episode_scores = data.get("episode scores", {})
                            if episode_scores.get("Lose") == 1:
                                dest_path = os.path.join(failures_dir, game, exp_dir.name, episode_dir.name)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copytree(episode_dir, dest_path, dirs_exist_ok=True)
                        except json.JSONDecodeError:
                            print("error: couldn't parse scores.json'")
    print(f"Data copied to: {failures_dir}")

def build_instances(dataset_path: str, instances_name: str):
    '''
    From a given dataset, build failed_instances.json, compatible with to_task_selector() - contains a list of rows with game, experiment, task_id columns
    '''
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    instances = []
    seen = set()
    for ep in data:
        meta_data = ep.get("meta", {})
        game = meta_data.get("game")
        experiment = meta_data.get("experiment")
        task_id = meta_data.get("task_id")
        key = (game, experiment, task_id)

        if key not in seen:
            seen.add(key)
            instance_data = {"game": game, "experiment": experiment, "task_id": task_id}
            instances.append(instance_data)

    output_path = os.path.join(os.path.dirname(dataset_path), f"{instances_name}.json")
    with open(output_path, "w") as fp:
        json.dump(instances, fp)

    print(f"Saved failed instances to {output_path}")



def cli(args: argparse.Namespace):
    if args.command_name == "list":
        if args.mode == "games":
            clem.list_games(args.selector, args.verbose)
        elif args.mode == "models":
            clem.list_models(args.verbose)
        elif args.mode == "backends":
            clem.list_backends(args.verbose)
        else:
            print(f"Cannot list {args.mode}. Choose an option documented at 'list -h'.")
    if args.command_name == "run":
        learner_spec = ModelSpec.from_string(args.learner)
        teacher_spec = ModelSpec.from_string(args.teacher) if args.teacher is not None else None
        train(args.file_path, learner_spec, teacher_spec, args.temperature, args.max_tokens)

    if args.command_name == "eval":
        model_spec = ModelSpec.from_string(args.model)
        gen_args = dict(temperature=args.temperature, max_tokens=args.max_tokens)
        evaluate(args.suite, model_spec, gen_args, args.results_dir, args.game, args.skip_gameplay)

    if args.command_name == "learn-from-failures":
        learner_spec = ModelSpec.from_string(args.learner)
        learner_name = learner_spec['model_name']
        #teacher_spec = ModelSpec.from_string(args.teacher)
        teacher_specs = [ModelSpec.from_string(t) for t in args.teacher.split()]
        #teacher_name = teacher_spec['model_name']
        gen_args = dict(temperature=args.temperature, max_tokens=args.max_tokens)

        results_dir_learner = f"./results_{learner_name}"
        failures_dir = f"./failures_{learner_name}/{learner_name}-t0.0"
        
        # create llama playthroughs
        clem.run("{'benchmark':['2.0']}", [learner_spec],
        gen_args=gen_args, results_dir=results_dir_learner)

        # score them because for some reason run doesn't do that
        clem.score("{'benchmark':['2.0']}", results_dir=results_dir_learner)

        # identify only failed instances from llama playthroughs and copy failed instances to new folder
        os.makedirs(failures_dir, exist_ok=True)
        collect_failures(results_dir_learner, failures_dir)
        print(f"Creating dataset from {failures_dir}...")
        create_conversational_dataset_for(failures_dir)
        print(f"Created dataset from {failures_dir}")
        print(f"Extracting tasks from dataset...")

        # make failed_instances.json
        build_instances(f"{failures_dir}/results.jsonl", "failed_instances")

        # run better model on these
        with open(os.path.join(os.path.dirname(f"./{failures_dir}/results.jsonl"), "failed_instances.json")) as f:
            dataset = json.load(f)
        task_selector = to_task_selector(dataset)
        results_dir_teacher = f"./results_teachers"
        for teacher in teacher_specs:
            teacher_name = teacher['model_name']
            print(f"Running better model {teacher_name} on failed instances...")
            clem.run("{'benchmark':['2.0']}", [teacher],
                    gen_args=gen_args, results_dir=results_dir_teacher, task_selector=task_selector)
            clem.score("{'benchmark':['2.0']}", results_dir=results_dir_teacher)
        print(f"Creating conversational dataset from {teacher_name} runs...")
        create_conversational_dataset_for(results_dir_teacher)
        
        # finetune learner model on successful best model runs
        print(f"Dataset created. Finetune learner model by running the command:")
        print(f"playpen run sft_trainer_lora.py -l {learner_name}")


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends. Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    train_parser = sub_parsers.add_parser("run")
    train_parser.add_argument("file_path", type=str,
                              help="The path to the trainer file to use for learning.")
    train_parser.add_argument("-l", "--learner", type=str,
                              help="The model name of the learner model (as listed by 'playpen list models').")
    train_parser.add_argument("-t", "--teacher", type=str, default=None,
                              help="The model name of the partner model (as listed by 'playpen list models')."
                                   "Optional, since non-interactive methods (like SFT) may not require a teacher model.",
                              required=False)
    train_parser.add_argument("-T", "--temperature", type=float, required=False, default=0.0,
                              help="The temperature used for generation. Should be the same as during training. "
                                   "Default: 0.0.")
    train_parser.add_argument("-L", "--max_tokens", type=int, required=False, default=300,
                              help="The token limit for generated responses. Should be the same as during training. "
                                   "Default: 300.")

    # Note: For now, we directly bound the eval to the playpen-data validate split.
    eval_parser = sub_parsers.add_parser("eval",
                                         description="Run the playpen eval pipelines to compute clem- and statscore.")
    eval_parser.add_argument("model", type=str,
                             help="The model name of the model to be evaluated (as listed by 'playpen list models').")
    eval_parser.add_argument("--suite", choices=["clem", "static", "all"], default="all",
                             nargs="?", type=str,
                             help="Choose which eval suites to run. Default: all")
    eval_parser.add_argument("-g", "--game", type=str,
                             help="A game selector e.g. a game name or a GameSpec-like JSON object given as a string.")
    eval_parser.add_argument("-r", "--results_dir", type=Path, default=get_default_results_dir(),
                             help="A relative or absolute path to a playpen-eval results directory. "
                                  "This is expected to be one level above 'clem' or 'static' results."
                                  "Default: playpen-eval/<timestamp>.")
    eval_parser.add_argument("--skip_gameplay", action="store_true",
                             help="Flag to skip gameplay and only calculate the clemscore for a given 'results_dir'."
                                  "Default: False. Only relevant for 'clem'.")
    eval_parser.add_argument("-T", "--temperature", type=float, default=0.0,
                             help="The temperature used for generation. Should be the same as during training. "
                                  "Default: 0.0.")
    eval_parser.add_argument("-L", "--max_tokens", type=int, default=300,
                             help="The token limit for generated responses. Should be the same as during training. "
                                  "Default: 300.")
    failure_parser = sub_parsers.add_parser("learn-from-failures", description="Play selected games with chosen model and gather failed episodes to make a new dataset from them")
    failure_parser.add_argument("--learner", type=str,
                             help="The model name of the model to be run (as listed by 'playpen list models').")
    failure_parser.add_argument("--teacher", type=str,
                             help="The model name of the model to learn from (as listed by 'playpen list models').")
    failure_parser.add_argument("-T", "--temperature", type=float, default=0.0,
                             help="The temperature used for generation. Should be the same as during training. "
                                  "Default: 0.0.")
    failure_parser.add_argument("-L", "--max_tokens", type=int, default=300,
                             help="The token limit for generated responses. Should be the same as during training. "
                                  "Default: 300.")
    
    # todo: add a 'playpen play' option to allow collection of new interaction data on the train split

    cli(parser.parse_args())


if __name__ == "__main__":
    main()
