import logging
import random
import time
from collections.abc import AsyncGenerator
from typing import override

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from agents.tic_tac_toe.tic_tac_toe.sub_agents.commentator.agent import (
    commentator_agent,
)
from agents.tic_tac_toe.tic_tac_toe.sub_agents.player.agent import (
    player1_agent,
    player2_agent,
)
from agents.tic_tac_toe.tic_tac_toe.tools import get_winner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Game(BaseAgent):
    """Custom agent for managing a game of Tic Tac Toe.

    This agent orchestrates a sequence of agents to manage the game state,
    validate player moves, and determine the outcome of the game.
    """

    # BaseAgent is a Pydantic model, so these are declared, validated fields
    # rather than plain attributes — see super().__init__() below.
    player1: LlmAgent
    player2: LlmAgent
    commentator: LlmAgent
    board: list[int]

    def __init__(
        self,
        name: str,
        player1: LlmAgent,
        player2: LlmAgent,
        commentator: LlmAgent,
    ) -> None:
        """Initialize the game."""
        board = [0] * 9

        # Pydantic's own __init__ does the real validation and assignment of
        # the fields declared above; without calling it, self.player1 etc.
        # would never actually get set on the instance.
        super().__init__(
            name=name,
            player1=player1,
            player2=player2,
            commentator=commentator,
            board=board,
            sub_agents=[player1, player2, commentator],
        )

    # Static-analysis-only marker confirming this overrides BaseAgent's method
    # of the same name; it has no effect at runtime.
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # ctx carries this invocation's session (and its shared state dict)
        # plus the session_service used to persist events.
        state = ctx.session.state

        # We can run this game agent in either LIVE or MOCK mode. In LIVE mode,
        # the game will be played out between the two players. In MOCK mode, we will
        # randomly select a winner without actually playing the game.
        # .get(key, default) avoids a KeyError when "mode" hasn't been set yet.
        mode = state.get("mode", "LIVE")
        logger.info("Starting game. Mode: %s", mode)

        result: str | None = None

        if mode == "MOCK":
            # If we are in MOCK mode, we will randomly select a winner without
            # actually playing the game.
            result = self._mock_play()
        else:
            # If we are in LIVE mode, we will play out the game between the two players
            # starting with player 1.
            current_player = 1

            # Let's store every available position on the board in the agent state.
            # We will use this list to validate each move made by a player.
            state["positions"] = [i for i, v in enumerate(self.board) if v == 0]

            # The game will continue until there are no more available positions
            # or one of the players has won the game.
            while len(state["positions"]) > 0:
                # Let's store the current board and the candidate moves in the agent
                # state. We'll access these values from other agents.
                state["board"] = self.board
                state["candidates"] = "\n".join(
                    f"{position}. ({position // 3 + 1}, {position % 3 + 1})"
                    for position in state["positions"]
                )

                logger.info("Player %d turn...", current_player)

                # Let's call the appropriate player agent to make a move.
                player = self.player1 if current_player == 1 else self.player2
                # run_async starts the player LlmAgent (an actual LLM call) and
                # returns an async generator of Events; async for streams them
                # in as they're produced instead of waiting for all of them.
                async for event in player.run_async(ctx):
                    yield event

                # After the player agent finishes, we can access its move from the
                # session state. ADK writes it into state["turn"] automatically
                # because the player agents declare output_key="turn".
                turn = state["turn"]
                position = turn["position"]

                logger.info(
                    "Player %d. Strategy: %s. Position: %s (%d, %d)",
                    current_player,
                    turn["strategy"],
                    position,
                    position // 3 + 1,
                    position % 3 + 1,
                )

                state["current_player"] = current_player
                state["position"] = f"({position // 3 + 1}, {position % 3 + 1})"

                # Let's update the board with the current player's move and update the
                # list of available positions.
                self.board[position] = current_player
                state["board"] = self.board
                state["positions"] = [i for i, v in enumerate(self.board) if v == 0]

                # Let's check whether the current player has won the game or if the
                # game is a draw.
                winner = get_winner(self.board)
                if winner == current_player:
                    result = f"PLAYER_{current_player}_WON"
                    logger.info("Player %d won.", current_player)
                elif winner == 0:
                    result = "DRAW"
                    logger.info("The game is a draw.")

                # Let's now call the commentator agent to provide some commentary on
                # the move that just happened. The commentator will use the "outcome"
                # to contextualize its commentary.
                state["outcome"] = result if result is not None else "ONGOING"
                # Same run_async/async for streaming pattern as the player call above.
                async for event in self.commentator.run_async(ctx):
                    yield event

                # If there is a winner or the game is a draw, we will stop the loop.
                if result is not None:
                    break

                # Finally, let's switch to the next player.
                current_player = 2 if current_player == 1 else 1

        # Before the agent finishes, we want to reset the board and have it ready
        # in case there's a new game.
        self.board = [0] * 9

        # Let's create an event to update the system state with the result of
        # the game.
        event = self._create_event(result)
        # A single one-shot async I/O call (unlike the streamed async for loops
        # above) — await pauses here until the write completes.
        await ctx.session_service.append_event(ctx.session, event)

    def _mock_play(self) -> str:
        winner = random.choice([0, 1, 2])
        return "DRAW" if winner == 0 else f"PLAYER_{winner}_WON"

    def _create_event(self, last_game_result: str) -> Event:
        """Create an event to update the system state at the end of the game."""
        state_delta = {
            "last_game_result": last_game_result,
        }

        return Event(
            invocation_id="game_ended",
            author=self.name,
            actions=EventActions(state_delta=state_delta),
            timestamp=time.time(),
        )


game_agent = Game(
    name="game",
    player1=player1_agent,
    player2=player2_agent,
    commentator=commentator_agent,
)
