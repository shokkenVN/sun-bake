import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("gui", "--gui"):
        from sun_bake.cli import main as cli_main
        cli_main(sys.argv[1:])
    else:
        from sun_bake.gui.main_window import run_gui
        sys.exit(run_gui())


if __name__ == "__main__":
    main()
