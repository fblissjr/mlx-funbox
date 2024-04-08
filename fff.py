import sys
import click
from mlx_lm import load, generate
import mlx.core as mx
from tool_utils import load_tools_from_file

DEFAULT_TEMP = 0.5
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0


@click.command()
@click.option(
    "-m",
    "--model",
    default="./mlx-community_c4ai-command-r-plus-4bit",
    help="Path to the model",
)
@click.option(
    "-t",
    "--max-tokens",
    type=int,
    default=256,
    help="Maximum number of tokens to generate",
)
@click.option("-s", "--stream", is_flag=True, default=True, help="Stream the output")
@click.option("--eos-token", default=None, help="End-of-sequence token")
@click.option(
    "--trust-remote-code", is_flag=True, default=False, help="Trust remote code"
)
@click.option("--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
@click.option("--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
@click.option(
    "--ignore-chat-template",
    is_flag=True,
    default=False,
    help="Use the raw prompt without the tokenizer's chat template",
)
@click.option(
    "--use-default-chat-template",
    is_flag=True,
    default=True,
    help="Use the default chat template",
)
@click.option(
    "--colorize",
    is_flag=True,
    default=False,
    help="Colorize output based on T[0] probability",
)
@click.option(
    "--use-tools",
    type=click.Path(exists=True),
    help="Enable tool use capabilities and specify the path to the JSON file containing tool definitions",
)
@click.argument("prompt", required=False)
def cli(
    model,
    max_tokens,
    stream,
    eos_token,
    trust_remote_code,
    temp,
    top_p,
    seed,
    ignore_chat_template,
    use_default_chat_template,
    colorize,
    use_tools,
    prompt,
):
    tools = []

    if not sys.stdin.isatty():
        prompt = prompt + sys.stdin.read() if prompt else sys.stdin.read()

    if use_tools:
        tools = load_tools_from_file(use_tools)

    generate_text(
        prompt,
        model,
        max_tokens,
        stream,
        eos_token,
        trust_remote_code,
        temp,
        top_p,
        seed,
        ignore_chat_template,
        use_default_chat_template,
        colorize,
        tools,
    )


def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.95:
        color = "white"
    elif t0 > 0.70:
        color = "green"
    elif t0 > 0.30:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)


def generate_text(
    prompt,
    model_path,
    max_tokens,
    stream,
    eos_token,
    trust_remote_code,
    temp,
    top_p,
    seed,
    ignore_chat_template,
    use_default_chat_template,
    colorize,
    tools,
):
    mx.random.seed(seed)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if trust_remote_code else None}
    if eos_token is not None:
        tokenizer_config["eos_token"] = eos_token

    model, tokenizer = load(model_path, tokenizer_config=tokenizer_config)

    if use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    if not ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif tools:
        conversation = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_tool_use_template(
            conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    formatter = colorprint_by_t0 if colorize else None

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=stream,
        temp=temp,
        top_p=top_p,
        formatter=formatter,
    )

    if not stream:
        print(response)


if __name__ == "__main__":
    cli()
