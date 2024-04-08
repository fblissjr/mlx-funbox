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
    "--use-tools",
    type=click.Path(exists=True),
    help="Enable tool use capabilities and specify the path to the JSON file containing tool definitions",
)
@click.option(
    "--colorize",
    is_flag=True,
    default=False,
    help="Colorize output based on T[0] probability",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode to print additional information",
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
    use_tools,
    prompt,
    colorize,
    debug,
):
    tools = []
    piped_content = None

    if not sys.stdin.isatty():
        piped_content = sys.stdin.read()
        prompt = prompt + piped_content if prompt else piped_content

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
        tools,
        colorize,
        debug,
        piped_content,
    )


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
    tools,
    colorize,
    debug,
    piped_content,
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

    if tools:
        conversation = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_tool_use_template(
            conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    if not ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if debug:
        print("Debug information:")
        print("Prompt:", prompt)
        print("Piped content:", piped_content)
        print("Tools:", tools)

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
