from pathlib import Path
import mooseherder as mh

def main() -> None:
    gmsh_path = Path().home() / 'moose-workdir/gmsh/bin/gmsh'
    gmsh_runner = mh.GmshRunner(gmsh_path)

    gmsh_input = Path('data/gmsh_test.geo')
    gmsh_runner.set_input_file(gmsh_input)

    gmsh_runner.run()


if __name__ == '__main__':
    main()