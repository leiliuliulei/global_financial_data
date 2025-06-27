from core.analyser import Cli
from rich.prompt import Prompt


def main():

    cli_window = Cli()
    cli_window.print_tree()

    while True:

        user_input = Prompt.ask(prompt='输入任意 行业序号 或 公司名称。输入 tree 展示行业列表，输入 exit 退出')

        if user_input.lower() == 'exit':
            break

        elif user_input.lower() == 'tree':
            cli_window.print_tree()

        else:
            cli_window.analyze(user_input)


if __name__ == '__main__':
    main()
