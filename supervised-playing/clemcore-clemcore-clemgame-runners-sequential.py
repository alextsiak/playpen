import logging
from typing import List

from tqdm import tqdm

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameBenchmarkCallbackList, GameInstanceIterator, GameStep

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def run(advisor_tok,
        advisor_model,
        game_benchmark: GameBenchmark,
        game_instance_iterator: GameInstanceIterator,
        player_models: List[Model],
        *,
        callbacks: GameBenchmarkCallbackList,
        ):
    callbacks.on_benchmark_start(game_benchmark)
    error_count = 0
    for experiment, game_instance in tqdm(game_instance_iterator, desc="Playing game instances"):
        try:
            game_master = game_benchmark.create_game_master(experiment, player_models)
            callbacks.on_game_start(game_master, game_instance)
            game_master.setup(**game_instance)
            done = False
            count = 0
            while not done:
                count += 1
                player, context = game_master.observe()
                
                if count == 10 or count == 25:
                    from clemcore.clemgame.callbacks.files import InteractionsFileSaver
                    _key = InteractionsFileSaver.to_key(game_master.game_spec.game_name, game_master.experiment["name"], game_instance["game_id"])
                    recorder = callbacks.callbacks[2]._recorders[_key]
                    judge_prompt = str(recorder.interactions)
                    judge_prompt = """You are an ADVISOR helping a player in a text-based game.
                            The Game Master has its own protocol. You must respect it.
                            
                            INPUT YOU SEE:
                            - The goal of the game.
                            - The official protocol description (supplied by the Game Master).
                            - A short history of recent turns (Game Master messages + Player responses).
                            - Feedback from the Game Master about failed actions.
                            
                            YOUR TASK:
                            - Analyze the player's recent responses and the Game Master's feedback.
                            - Identify concrete mistakes the player made (wrong action format, acting on nonexistent objects, using invalid targets, etc.).
                            - For each mistake, also show the correct action that should have been used instead.
                            - Suggest 2–3 clear recommendations that will help the player avoid repeating those mistakes.
                            - Do not repeat the exact same advice wording across turns; highlight new aspects or rephrase if needed.
                            - Propose ONE valid next action that follows the protocol and moves toward the goal, based on the most recent Game Master observation.
                            
                            OUTPUT FORMAT (strict):
                            Mistakes noticed:
                            - <bullet list of mistakes, each followed by a suggested correction>
                            
                            Advice:
                            1. <recommendation>
                            2. <recommendation>
                            3. <recommendation> (if needed, phrased differently from past advice)
                            
                            Best next action:
                            > <single valid action according to the current protocol>""" + "So the recent responses so far are: " + judge_prompt 

                    from transformers import pipeline
                    
                    advisor = pipeline(
                        "text-generation",
                        model=advisor_model,
                        tokenizer=advisor_tok,
                        max_new_tokens=196,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        repetition_penalty=1.05,
                    )
                    
                    def get_advice(prompt: str) -> str:
                        out = advisor(prompt)[0]["generated_text"]
                        # Some models echo the prompt; remove it if needed:
                        return out[len(prompt):].strip()

                    advisor_recommendation = get_advice(judge_prompt)
                    context["content"] = "My advice for you so far: "+advisor_recommendation+" So now continue: "+context["content"]
                                    
                response = player(context)
                done, info = game_master.step(response)
                game_step = GameStep(context, response, done, info)
                callbacks.on_game_step(game_master, game_instance, game_step)
            callbacks.on_game_end(game_master, game_instance)
        except Exception:  # continue with other instances if something goes wrong
            message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
            module_logger.exception(message)
            error_count += 1
    if error_count > 0:
        stdout_logger.error(
            f"{game_benchmark.game_name}: '{error_count}' exceptions occurred: See clembench.log for details.")
    callbacks.on_benchmark_end(game_benchmark)
